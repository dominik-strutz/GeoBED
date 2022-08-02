import torch
from torch import nn

import numpy as np
import pyro
from tqdm import tqdm
import random

import matplotlib.pyplot as plt

import pyro.distributions as dist
from torch.distributions import constraints
import pyro.distributions.transforms as transforms
import torch.distributions.transforms as torch_transforms

import pyro


def var_post(self, hdf, var_guide, n_steps,
             n_samples, n_final_samples=-1,
             stoch_sampling=True, n_stochastic=None,
             optim=None, scheduler=None,
             N_max=-1, batched=False, guide_args={},
             plot_loss=True, return_guide=False,
             design_selection=None,
             **kwargs):
    
    n_samples       = n_samples       if not n_samples       == -1 else self.n_prior
    n_final_samples = n_final_samples if not n_final_samples == -1 else self.n_prior
    
    if n_samples  > self.n_prior: raise ValueError(
        'Not enough prior samples, choose different n_samples!')
    if n_final_samples > self.n_prior: raise ValueError(
        'Not enough prior samples, choose different n_final_samples!')
    
    if stoch_sampling:
        if n_stochastic >= n_samples: raise ValueError(
            'No stochastic sampling possible when all samples are used')
    
    # set costum optimizer or fall back to default if none is set
    if optim is None:
        def optimizer_constructor(guide):
            return torch.optim.Adam(guide.parameters(), lr=1e-2)
    else:
        optimizer_constructor = optim
        
    # set costum scheduler or fall back to default if none is set
    if scheduler is not None:
        scheduler_constructor  = scheduler['scheduler']
        scheduler_n_iterations = scheduler['n_iterations']
    
    try:
        prior_ent = self.model_prior.entropy()
    except AttributeError:
        raise NotImplementedError('Monte Carlo estimate of prior entropy not yet implemented')
    
    # safe output for each design step to those lists #TODO make this mor sophisticated 
    eig_list = []
    if return_guide:
        var_list = []

    model_space = torch.tensor(hdf['prior'][:n_samples])

    for i, design_i in tqdm(enumerate(hdf['design']),
                            total=self.n_design,
                            disable=self.disable_tqdm):
        
        if design_selection is not None:
            if i not in design_selection:
                continue

        # set random set so each design point has same noise realisation
        pyro.set_rng_seed(0)
        
        samples = torch.tensor(hdf['data'][:n_samples, i])         
        
        if var_guide == 'mdn':
            guide = MDN_guide(samples, model_space, **guide_args)
        elif var_guide == 'cpl_spline_nf':
            guide = Cond_Spline_NF(samples, model_space, **guide_args)
        elif type(var_guide) == str:
            raise NotImplementedError('Guide not implemented')
        
        optimizer = optimizer_constructor(guide)
        scheduler = scheduler_constructor(optimizer)
        
        losses = []

        for step in (pbar := tqdm(range(n_steps), total=n_steps, disable=True, leave=False)):
            
            optimizer.zero_grad()
            
            if stoch_sampling:
                random_indices = random.sample(range(n_samples), n_stochastic)
                x = self.data_likelihood(samples[random_indices]).sample([1, ]).flatten(start_dim=0, end_dim=1)
                ln_p_x2_given_x1 = guide.log_prob(model_space[random_indices], x)
            else:
                x = self.data_likelihood(samples).sample([1, ]).flatten(start_dim=0, end_dim=1)    
                ln_p_x2_given_x1 = guide.log_prob(model_space, x)
             
            loss = -(ln_p_x2_given_x1).mean()
            loss.backward() 
            optimizer.step()
            guide.clear_cache()

            if scheduler is not None:
                if step % scheduler_n_iterations == 0:
                    scheduler.step()

            losses.append(loss.detach().item())

            pbar.set_description(f"Loss: {loss.detach().item():.2e}, lr: {optimizer.param_groups[0]['lr']:.2e}")
        
        likeliehood = self.data_likelihood(samples[:n_samples], design_i)
        final_samples = likeliehood.sample([1, ]).flatten(start_dim=0, end_dim=1)
        
        #TODO: should be the same as loss without minus sign use directly for faster evaluations
        marginal_lp = guide.log_prob(model_space, final_samples)
        
        eig = prior_ent - (-marginal_lp).sum(0) / n_samples 
        
        eig_list.append(eig.detach().item())                   
        
        if return_guide:
            var_list.append(guide)
        
        if plot_loss:
            color = plt.cm.viridis(np.linspace(0, 1, self.n_design))
            plt.plot(losses, color=color[i]) 

    if return_guide:
        return np.array(eig_list), return_guide
    else:
        return np.array(eig_list)
    
    
class VarPost_Base(torch.nn.Module):
    def __init__(self):
        super().__init__()   
        

class MDN_guide(VarPost_Base):
    
    def __init__(self, data_samples, model_sample, K=3):
        super(MDN_guide, self).__init__()
        
        self.design_dim = data_samples.shape[-1]
        self.model_dim = model_sample.shape[-1]
        self.K = K
        
        self.nn_N = 64

        self.ds_means = data_samples.mean(dim=0, keepdims=True)
        self.ds_stds  = data_samples.std(dim=0, keepdims=True)
        
        self.ms_means = model_sample.mean(dim=0, keepdims=True)
        self.ms_stds  = model_sample.std(dim=0, keepdims=True)
        
        self.z_1 = nn.Linear(self.design_dim , self.nn_N)
        self.z_2 = nn.Linear(self.nn_N, self.nn_N)
        # self.z_3 = nn.Linear(self.nn_N, self.nn_N)

        self.h_alpha = nn.Linear(self.nn_N, self.K)
        self.h_mu =    nn.Linear(self.nn_N, self.K*self.model_dim)
        self.h_sigma = nn.Linear(self.nn_N, self.K*self.model_dim)
        
    def evaluate(self, data_samples):
        
        batch_dim = data_samples.shape[:-1]
        
        x = data_samples.clone()
        
        x -= self.ds_means
        x /= self.ds_stds        

        z_h = torch.tanh(self.z_1(data_samples))
        z_h = torch.tanh(self.z_2(z_h))
        # z_h = torch.tanh(self.z_3(z_h))
        
        alpha = nn.functional.log_softmax(self.h_alpha(z_h), -1)
        mu    = self.h_mu(z_h)
        sigma = torch.nn.functional.elu(self.h_sigma(z_h)) + 1.0 + 1e-15

        alpha = alpha.reshape((*batch_dim, self.K))
        mu    = mu.reshape((*batch_dim, self.K, self.model_dim))
        sigma = sigma.reshape((*batch_dim, self.K, self.model_dim))
        
        mix = dist.Categorical(logits=alpha)
        comp = dist.Normal(mu , sigma).to_event(1)
        dit = dist.MixtureSameFamily(mix, comp)
        
        normalizing = torch_transforms.AffineTransform(
            self.ms_means, self.ms_stds, event_dim=1)
        
        dit = dist.TransformedDistribution(dit, normalizing)
        
        # dit = dist.MixtureOfDiagNormals(locs=mu,
        #                                 coord_scale=sigma,
        #                                 component_logits=alpha)
        
        return alpha, mu, sigma, dit
        
    def log_prob(self, model_sample, data_samples):
        
        _, _, _, dit = self.evaluate(data_samples)           
        
        return dit.log_prob(model_sample)
    
    def clear_cache(self):
        pass
    
    def sample(self, data_samples, size):
        
        _, _, _, dit = self.evaluate(data_samples)
        sample = dit.sample(size)
        return sample
    
    def get_mix(self, data_samples):
        alpha, _, _, _ = self.evaluate(data_samples)
        return alpha
    
    def get_mean(self, data_samples):
        _, mu, _, _ = self.evaluate(data_samples)
        return mu
    
    def get_sigma(self, data_samples):
        _, _, sigma, _ = self.evaluate(data_samples)
        return sigma


class Cond_Spline_NF(VarPost_Base):
    
    def __init__(self, data_samples, model_sample, K=3):
        super(Cond_Spline_NF, self).__init__()
        
        self.design_dim = data_samples.shape[-1]
        self.model_dim  = model_sample.shape[-1]
        
        self.ds_means = data_samples.mean(dim=0, keepdims=True)
        self.ds_stds  = data_samples.std(dim=0, keepdims=True)
        
        self.ms_means = model_sample.mean(dim=0, keepdims=True)
        self.ms_stds  = model_sample.std(dim=0, keepdims=True)
        
        dist_base_m = dist.Normal(torch.zeros(self.model_dim),
                                  torch.ones(self.model_dim))
        
        self.x2_transform_1 = transforms.conditional_spline(
            self.model_dim, context_dim=self.design_dim, count_bins=32)

        ms_normalizing = torch_transforms.AffineTransform(
            self.ms_means, self.ms_stds, event_dim=1)
        
        self.dist_x2_given_x1 = dist.ConditionalTransformedDistribution(dist_base_m, [self.x2_transform_1, ms_normalizing])
        
    def log_prob(self, model_space, data_samples):
        
        dit = self.dist_x2_given_x1.condition(data_samples)       
        
        return dit.log_prob(model_space)

    def parameters(self):
        return torch.nn.ModuleList(
            [self.x2_transform_1,]).parameters()

    def clear_cache(self):
        self.dist_x2_given_x1.clear_cache()
    
    def sample(self, data_samples, size):
        
        dit = self.dist_x2_given_x1.condition(data_samples.unsqueeze(-1))
        sample = dit.sample(size)
        return sample

