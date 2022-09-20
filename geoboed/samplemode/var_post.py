from pyexpat import model
import random

import torch
from torch import nn

import numpy as np
import pyro
from tqdm import tqdm

import pyro.distributions as dist
import pyro.distributions.transforms as transforms
import torch.distributions.transforms as torch_transforms


def var_post(self, dataframe, design_list,
             var_guide, n_steps,
             n_samples, n_final_samples=-1,
             stoch_sampling=True, n_stochastic=None,
             optim=None, scheduler=None,
             batched=False, guide_args={},
             return_dict=False, preload_samples=True,
             interrogation_mapping=None, 
             disable_tqdm=False, set_rseed=True,
             model_prior=None,
             **kwargs):
    
    n_samples       = n_samples       if not n_samples       == -1 else self.n_prior
    n_final_samples = n_final_samples if not n_final_samples == -1 else n_samples
    
    if n_samples  > self.n_prior: raise ValueError(
        'Not enough prior samples, choose different n_samples!')
    if n_final_samples > self.n_prior: raise ValueError(
        'Not enough prior samples, choose different n_final_samples!')    
    if stoch_sampling:
        if n_stochastic >= n_samples: raise ValueError(
            'No stochastic sampling possible when all samples are used')
    
    n_max = max(n_samples, n_final_samples)
    
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
    
    if var_guide == 'mdn':
        guide_template = MDN_guide
    elif var_guide == 'mdn_cov':
        guide_template = MDN_guide_cov
    elif var_guide == 'cond_spline_nf':
        guide_template = Cond_Spline_NF
    elif type(var_guide) == str:
        raise NotImplementedError('Guide not implemented')

    if model_prior is not None:
        try:
            prior_ent = model_prior.entropy()
        except AttributeError:
            raise NotImplementedError('Monte Carlo estimate of prior entropy not yet implemented')
    else:
        raise NotImplementedError('Monte Carlo estimate of prior entropy not yet implemented')

    
    model_space = torch.tensor(dataframe['prior'][:n_max]).float()
    
    if batched:
        raise NotImplementedError()
    
    else:
        eig_list = []
        if return_dict:
            guide_collection = []
        losses_collection = []

        if preload_samples:
            pre_samples = dataframe['data'][:n_max]

        for design_i in tqdm(design_list, total=len(design_list), disable=disable_tqdm):

            if set_rseed:
                pyro.set_rng_seed(0)
                torch.manual_seed(0)
                                    
            # .tolist necessary to keep data dimension even for one reciver designs
            if preload_samples:
                samples = torch.tensor(pre_samples[ :n_max, design_i.tolist()]).float()
            else:
                samples = torch.tensor(dataframe['data'][ :n_max, design_i.tolist()]).float()
            
            if interrogation_mapping is not None:
                model_space_samples = interrogation_mapping(model_space[:n_max])
            else:
                model_space_samples = model_space[:n_max]
            
            guide = guide_template(samples, model_space_samples, guide_args)
            optimizer = optimizer_constructor(guide)
            scheduler = scheduler_constructor(optimizer)
            
            losses = []

            for step in (tqdm(range(n_steps), total=n_steps, disable=True, leave=False)):
                
                optimizer.zero_grad()
                
                if stoch_sampling:            
                    random_indices = random.sample(range(n_samples), n_stochastic)                    
                    x = self.data_likelihood(samples[random_indices], self.get_designs()[design_i]).sample([1, ]).flatten(start_dim=0, end_dim=1)
                    
                    ln_p_x2_given_x1 = guide.log_prob(model_space_samples[random_indices], x)
                else:
                    x = self.data_likelihood(samples[:n_samples], self.get_designs()[design_i]).sample([1, ]).flatten(start_dim=0, end_dim=1)    
                    ln_p_x2_given_x1 = guide.log_prob(model_space_samples[:n_samples], x)
                
                loss = -(ln_p_x2_given_x1).mean()
                loss.backward() 
                optimizer.step()
                guide.clear_cache()

                if scheduler is not None:
                    if step % scheduler_n_iterations == 0:
                        scheduler.step()

                if return_dict: losses.append(loss.detach().item())
                 
                # pbar.set_description(f"Loss: {loss.detach().item():.2e}, lr: {optimizer.param_groups[0]['lr']:.2e}")

            losses_collection.append(losses)
            if return_dict:
                guide_collection.append(guide)
            
            likeliehood = self.data_likelihood(samples[:n_final_samples], self.get_designs()[design_i])
            samples = likeliehood.sample([1, ]).flatten(start_dim=0, end_dim=1)
            
            #TODO: should be the same as loss without minus sign use directly for faster evaluations
            marginal_lp = guide.log_prob(model_space_samples[:n_final_samples], samples)
                        
            eig = prior_ent - (-marginal_lp).sum(0) / n_final_samples 
            
            eig_list.append(eig.detach().item())                   
            
            
        if return_dict:
            output_dict = {
                'var_guide_name':  var_guide,
                'var_guide':       guide_collection,
                'n_steps':         n_steps,
                'n_samples':       n_samples,
                'n_final_samples': n_final_samples,
                'stoch_sampling':  stoch_sampling,
                'n_stochastic':    n_stochastic,
                'optim':           optim,
                'scheduler':       scheduler,
                'batched':         batched,
                'guide_args':      guide_args,
                'losses':          np.array(losses_collection).T,
                }
                            
            return np.array(eig_list), output_dict
        else:
            return np.array(eig_list), None
        

class MDN_guide(torch.nn.Module):
    
    def __init__(self, data_samples, model_sample, guide_args):
        super(MDN_guide, self).__init__()
        
        self.design_dim = data_samples.shape[-1]
        self.model_dim = model_sample.shape[-1]

        self.K              = guide_args['K'] if 'K' in guide_args else 2
        self.nn_layers_args = guide_args['nn_layers'] if 'nn_layers' in guide_args else [64]

        self.ds_means = data_samples.mean(dim=0)
        self.ds_stds  = data_samples.std(dim=0)
        
        self.ms_means = model_sample.mean(dim=0)
        self.ms_stds  = model_sample.std(dim=0)
        
        self.nn_layer_list = [nn.Linear(self.design_dim , self.nn_layers_args[0]), nn.Tanh()]
        for i_layer in range(len(self.nn_layers_args)-1):

            self.nn_layer_list.append(
                nn.Linear(self.nn_layers_args[i_layer] , self.nn_layers_args[i_layer+1])
                )
            self.nn_layer_list.append(
                nn.Tanh()
                )

        self.nn = nn.Sequential(*self.nn_layer_list)

        self.h_alpha = nn.Linear(self.nn_layers_args[-1], self.K)
        self.h_mu =    nn.Linear(self.nn_layers_args[-1], self.K*self.model_dim)
        self.h_sigma = nn.Linear(self.nn_layers_args[-1], self.K*self.model_dim)
        
    def evaluate(self, data_samples):
        
        batch_dim = data_samples.shape[0]
               
        x = data_samples.clone()
                
        x -= self.ds_means
        x /= self.ds_stds        
        
        z_h = self.nn(x)
        
        alpha = nn.functional.log_softmax(self.h_alpha(z_h), -1)
        mu    = self.h_mu(z_h)
        sigma = torch.nn.functional.elu(self.h_sigma(z_h)) + 1.0 + 1e-15

        alpha = alpha.reshape((batch_dim, self.K))
        mu    = mu.reshape((batch_dim, self.K, self.model_dim))
        sigma = sigma.reshape((batch_dim, self.K, self.model_dim))
                
        mix = dist.Categorical(logits=alpha)
        comp = dist.Normal(mu , sigma).to_event(1)
        dit = dist.MixtureSameFamily(mix, comp)
        
        normalizing = torch_transforms.AffineTransform(
            self.ms_means, self.ms_stds, event_dim=1)
                
        dit = dist.TransformedDistribution(dit, normalizing)
                
        return alpha, mu, sigma, dit
        
    def log_prob(self, model_samples, data_samples):

        _, _, _, dit = self.evaluate(data_samples)           
        
                
        return dit.log_prob(model_samples)
    
    def clear_cache(self):
        pass
    
    def sample(self, data_samples):
        #TODO make sampling a bit more transparent with how many samples should be drawn from each posterior individually

        _, _, _, dit = self.evaluate(data_samples)
        
        sample = dit.sample([1,]).flatten(start_dim=0, end_dim=1)
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
    
    
    
class MDN_guide_cov(torch.nn.Module):
    
    def __init__(self, data_samples, model_sample, guide_args):
        super(MDN_guide_cov, self).__init__()
        
        self.design_dim = data_samples.shape[-1]
        self.model_dim = model_sample.shape[-1]

        self.K              = guide_args['K'] if 'K' in guide_args else 2
        self.nn_layers_args = guide_args['nn_layers'] if 'nn_layers' in guide_args else [64]

        self.ds_means = data_samples.mean(dim=0)
        self.ds_stds  = data_samples.std(dim=0)
        
        self.ms_means = model_sample.mean(dim=0)
        self.ms_stds  = model_sample.std(dim=0)
                
        self.nn_layer_list = [nn.Linear(self.design_dim , self.nn_layers_args[0]), nn.Tanh()]
        for i_layer in range(len(self.nn_layers_args)-1):

            self.nn_layer_list.append(
                nn.Linear(self.nn_layers_args[i_layer] , self.nn_layers_args[i_layer+1])
                )
            self.nn_layer_list.append(
                nn.Tanh()
                )

        self.nn = nn.Sequential(*self.nn_layer_list)

        self.h_alpha = nn.Linear(self.nn_N, self.K)
        self.h_mu    = nn.Linear(self.nn_N, self.K*self.model_dim)
        self.h_cov   = nn.Linear(self.nn_N, self.K*self.model_dim*self.model_dim)
        
    def evaluate(self, data_samples):
        
        batch_dim = data_samples.shape[0]
               
        x = data_samples.clone()
                
        x -= self.ds_means
        x /= self.ds_stds        

        z_h = torch.tanh(self.z_1(x))
        z_h = torch.tanh(self.z_2(z_h))
        # z_h = torch.tanh(self.z_3(z_h))
        
        alpha = nn.functional.log_softmax(self.h_alpha(z_h), -1)
        mu    = self.h_mu(z_h)
        cov   = torch.nn.functional.elu(self.h_cov(z_h)) + 1.0 + 1e-15

        alpha = alpha.reshape((batch_dim, self.K))
        mu    = mu.reshape((batch_dim, self.K, self.model_dim))
        cov   = cov.reshape((batch_dim, self.K, self.model_dim, self.model_dim))
                
        mix = dist.Categorical(logits=alpha)
        
        # ensure positive-definiteness
        cov = torch.matmul(cov, torch.transpose(cov, -1, -2)) + \
            torch.diag_embed(torch.ones(batch_dim, self.K, self.model_dim)) * 1e-10
        comp = dist.MultivariateNormal(mu, covariance_matrix=cov)
        
        dit = dist.MixtureSameFamily(mix, comp)
        
        normalizing = torch_transforms.AffineTransform(
            self.ms_means, self.ms_stds, event_dim=1)
                
        dit = dist.TransformedDistribution(dit, normalizing)
        
        return alpha, mu, cov, dit
        
    def log_prob(self, model_samples, data_samples):

        _, _, _, dit = self.evaluate(data_samples)           
        
                
        return dit.log_prob(model_samples)
    
    def clear_cache(self):
        pass
    
    def sample(self, data_samples):
        #TODO make sampling a bit more transparent with how many samples should be drawn from each posterior individually

        _, _, _, dit = self.evaluate(data_samples)
        
        sample = dit.sample([1,]).flatten(start_dim=0, end_dim=1)
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


class Cond_Spline_NF(torch.nn.Module):
    
    def __init__(self, data_samples, model_sample,guide_args={}):
        super(Cond_Spline_NF, self).__init__()
        
        self.design_dim = data_samples.shape[-1]
        self.model_dim  = model_sample.shape[-1]
        
        self.ds_means = data_samples.mean((0, 1))
        self.ds_stds  = data_samples.std((0, 1))
        
        self.ms_means = model_sample.mean(dim=0)
        self.ms_stds  = model_sample.std(dim=0)
        
        dist_base_m = dist.Normal(torch.zeros(self.model_dim),
                                  torch.ones(self.model_dim))
        
        self.transform_list = []

        if 'flows' in guide_args: 
            for i_f, flow in enumerate(guide_args['flows']):
                self.transform_list.append(flow(self.desing_dim, context_dim=self.design_dim, **guide_args['flow_args'][i_f]))
        else:
           self.transform_list.append(
               transforms.conditional_spline(self.model_dim, context_dim=self.design_dim, count_bins=32))

        self.transform_list.append(
            torch_transforms.AffineTransform(self.ms_means, 3*self.ms_stds, event_dim=1))
        
        self.dist_x2_given_x1 = dist.ConditionalTransformedDistribution(dist_base_m, self.transform_list)
        
    def log_prob(self, model_space, data_samples):
        
        x = data_samples.clone()
        
        x -= self.ds_means
        x /= self.ds_stds  
        
        dit = self.dist_x2_given_x1.condition(x)       
        
        return dit.log_prob(model_space)

    def parameters(self):
        return torch.nn.ModuleList(
           self.transform_list).parameters()

    def clear_cache(self):
        self.dist_x2_given_x1.clear_cache()
    
    def sample(self, data_samples):
        #TODO make sampling a bit more transparent with how many samples should be drawn from each posterior individually
        
        dit = self.dist_x2_given_x1.condition(data_samples)
                
        sample = dit.sample(data_samples.shape[:-1])
        return sample

