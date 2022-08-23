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
             model_prior=None,
             optim=None, scheduler=None,
             batched=False, guide_args={},
             return_dict=False, preload_samples=True,
             interrogation_mapping=None,
             **kwargs):
    
    n_samples       = n_samples       if not n_samples       == -1 else self.n_prior
    n_final_samples = n_final_samples if not n_final_samples == -1 else n_samples
    
    if n_samples  > self.n_prior: raise ValueError(
        'Not enough prior samples, choose different n_samples!')
    if n_final_samples > self.n_prior: raise ValueError(
        'Not enough prior samples, choose different n_final_samples!')    
    if n_final_samples > n_samples: raise ValueError(
        'n_final_samples cant be larger than n_samples')
    
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

    
    model_space = torch.tensor(dataframe['prior'][:n_samples]).float()
    
    if batched:
        raise NotImplementedError()
        # #TODO: Cant train the same nn for all designs, obiously ...
        
        # if design_selection:
        #     samples = torch.tensor(dataframe['data'][ :n_samples, design_selection])
        #     model_space = torch.tensor(dataframe['prior'][:n_samples])

        # else:
        #     samples = torch.tensor(dataframe['data'][ :n_samples, :])
        #     model_space = torch.tensor(dataframe['prior'][:n_samples])

        # # # set random set so each design point has same noise realisation
        # # pyro.set_rng_seed(0)
                
        # if var_guide == 'mdn':
        #     guide = MDN_guide_batched(samples, model_space, **guide_args)
        # elif type(var_guide) == str:
        #     raise NotImplementedError('Guide not implemented')
        
        # optimizer = optimizer_constructor(guide)
        # if scheduler is not None:
        #     scheduler = scheduler_constructor(optimizer)        
        # losses = []

        # for step in (pbar := tqdm(range(n_steps), total=n_steps, disable=False, leave=False)):
            
        #     optimizer.zero_grad()
            
        #     if stoch_sampling:
        #         random_indices = random.sample(range(n_samples), n_stochastic)
        #         x = self.data_likelihood(samples[random_indices]).sample([1, ]).flatten(start_dim=0, end_dim=1)

        #         ln_p_x2_given_x1 = guide.log_prob(model_space[random_indices], x)
        #     else:
        #         x = self.data_likelihood(samples).sample([1, ]).flatten(start_dim=0, end_dim=1)    
        #         ln_p_x2_given_x1 = guide.log_prob(model_space, x)
            
        #     loss = -(ln_p_x2_given_x1.mean(dim=0))

        #     loss.backward(torch.ones_like(loss)) 
        #     optimizer.step()
        #     guide.clear_cache()

        #     if scheduler is not None:
        #         if step % scheduler_n_iterations == 0:
        #             scheduler.step()

        #     losses.append(loss.detach().tolist())

        #     pbar.set_description(f"Loss: {loss.sum().detach().item():.2e}, lr: {optimizer.param_groups[0]['lr']:.2e}")
            
        # if design_selection:
        #     final_samples = torch.tensor(dataframe['data'][ :n_final_samples, design_selection])
        #     likeliehood = self.data_likelihood(final_samples, dataframe['design'][design_selection])
        # else:
        #     final_samples = torch.tensor(dataframe['data'][ :n_final_samples, :])
        #     likeliehood = self.data_likelihood(final_samples, dataframe['design'][:])
        
        # model_space = torch.tensor(dataframe['prior'][:n_final_samples])

        # final_samples = likeliehood.sample([1, ]).flatten(start_dim=0, end_dim=1)
        
        # #TODO: should be the same as loss without minus sign use directly for faster evaluations
        # marginal_lp = guide.log_prob(model_space, final_samples)
        
        # eig = prior_ent + -(-marginal_lp).sum(0) / n_samples 
        
        # eig_list = eig.detach().tolist()                  
        
        # if return_guide:
        #     var_list = guide
        
        # if plot_loss:
        #     color = plt.cm.viridis(np.linspace(0, 1, self.n_design))
        #     plt.plot(losses) 
    
    else:
        eig_list = []
        if return_dict:
            guide_collection = []
        losses_collection = []

        if preload_samples:
            pre_samples = dataframe['data'][:n_final_samples]

        for design_i in tqdm(design_list, total=len(design_list), disable=self.disable_tqdm):

            pyro.set_rng_seed(0)
                        
            # .tolist necessary to keep data dimension even for one reciver designs
            if preload_samples:
                samples = torch.tensor(pre_samples[ :n_final_samples, design_i.tolist()]).float()
            else:
                samples = torch.tensor(dataframe['data'][ :n_final_samples, design_i.tolist()]).float()
            
            if interrogation_mapping is not None:
                model_space_samples = interrogation_mapping(model_space)
            else:
                model_space_samples = model_space[:]
            
            guide = guide_template(samples, model_space_samples, **guide_args)
            optimizer = optimizer_constructor(guide)
            scheduler = scheduler_constructor(optimizer)
            
            losses = []

            for step in (tqdm(range(n_steps), total=n_steps, disable=True, leave=False)):
                
                optimizer.zero_grad()
                
                if stoch_sampling:
                                        
                    random_indices = random.sample(range(n_samples), n_stochastic)                    
                    x = self.data_likelihood(samples[random_indices]).sample([1, ]).flatten(start_dim=0, end_dim=1)
                    
                    ln_p_x2_given_x1 = guide.log_prob(model_space_samples[random_indices], x)
                else:
                    x = self.data_likelihood(samples[:n_samples]).sample([1, ]).flatten(start_dim=0, end_dim=1)    
                    ln_p_x2_given_x1 = guide.log_prob(model_space_samples, x)
                
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
            
            likeliehood = self.data_likelihood(samples, design_i)
                        
            samples = likeliehood.sample([1, ]).flatten(start_dim=0, end_dim=1)
            
            #TODO: should be the same as loss without minus sign use directly for faster evaluations
            marginal_lp = guide.log_prob(model_space_samples, samples)
            
            post_samples = guide.sample(samples)
            
            # import matplotlib.pyplot as plt
            # plt.scatter(-post_samples[:, 0], post_samples[:, 1])
            # plt.show()
            
            # print(guide.sample(samples).shape)
            # print(guide.get_mean(samples).shape)

            # print(guide.sample(samples))
            
            eig = prior_ent - (-marginal_lp).sum(0) / n_samples 
            
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
    
    def __init__(self, data_samples, model_sample, K=3):
        super(MDN_guide, self).__init__()
        
        self.design_dim = data_samples.shape[-1]
        self.model_dim = model_sample.shape[-1]
        self.K = K
        
        self.nn_N = 128

        self.ds_means = data_samples.mean(dim=0)
        self.ds_stds  = data_samples.std(dim=0)
        
        self.ms_means = model_sample.mean(dim=0)
        self.ms_stds  = model_sample.std(dim=0)
                
        self.z_1 = nn.Linear(self.design_dim , self.nn_N)
        self.z_2 = nn.Linear(self.nn_N, self.nn_N)
        # self.z_3 = nn.Linear(self.nn_N, self.nn_N)

        self.h_alpha = nn.Linear(self.nn_N, self.K)
        self.h_mu =    nn.Linear(self.nn_N, self.K*self.model_dim)
        self.h_sigma = nn.Linear(self.nn_N, self.K*self.model_dim)
        
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
    
    def __init__(self, data_samples, model_sample, K=3):
        super(MDN_guide_cov, self).__init__()
        
        self.design_dim = data_samples.shape[-1]
        self.model_dim = model_sample.shape[-1]
        self.K = K
        
        self.nn_N = 128

        self.ds_means = data_samples.mean(dim=0)
        self.ds_stds  = data_samples.std(dim=0)
        
        self.ms_means = model_sample.mean(dim=0)
        self.ms_stds  = model_sample.std(dim=0)
                
        self.z_1 = nn.Linear(self.design_dim , self.nn_N)
        self.z_2 = nn.Linear(self.nn_N, self.nn_N)
        # self.z_3 = nn.Linear(self.nn_N, self.nn_N)

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
        
        # ensure positive-definiteness (bmm because batch of matrix is used)
        cov = torch.matmul(cov, torch.transpose(cov, -1, -2)) + \
            torch.diag_embed(torch.ones(batch_dim, self.K, self.model_dim)) * 1e-5
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


# class MDN_guide_batched(VarPost_Base):
    
    # def __init__(self, data_samples, model_sample, K=3):
    #     super(MDN_guide_batched, self).__init__()
        
    #     self.K = K
    #     self.design_dim = data_samples.shape[-2]

    #     self.data_dim = data_samples.shape[-1]
    #     self.model_dim = model_sample.shape[-1]

    #     self.nn_N = 128

    #     self.ds_means = data_samples.mean(dim=0)
    #     self.ds_stds  = data_samples.std(dim=0)
                
    #     self.ms_means = model_sample.mean(dim=0, keepdim=True)
    #     self.ms_stds  = model_sample.std(dim=0, keepdim=True)
        
    #     self.z_1 = nn.Linear(self.data_dim, self.nn_N)
    #     self.z_2 = nn.Linear(self.nn_N, self.nn_N)
    #     # self.z_3 = nn.Linear(self.nn_N, self.nn_N)

    #     self.h_alpha = nn.Linear(self.nn_N, self.K)
    #     self.h_mu =    nn.Linear(self.nn_N, self.K*self.model_dim)
    #     self.h_sigma = nn.Linear(self.nn_N, self.K*self.model_dim)
        
    # def evaluate(self, data_samples):
        
    #     batch_dim = data_samples.shape[:-1]
        
    #     # print(batch_dim       
        
    #     # print(x.shape)
    #     # print(self.ds_means.shape)
    #     # print(self.ds_stds.shape)
    #     data_samples = data_samples.unsqueeze(1)
        
    #     data_samples -= self.ds_means
    #     data_samples /= self.ds_stds        
        
    #     z_h = torch.tanh(self.z_1(data_samples))
    #     z_h = torch.tanh(self.z_2(z_h))
    #     # z_h = torch.tanh(self.z_3(z_h))
        
    #     alpha = nn.functional.log_softmax(self.h_alpha(z_h), dim=-1)
    #     mu    = self.h_mu(z_h)
    #     sigma = torch.nn.functional.elu(self.h_sigma(z_h)) + 1.0 + 1e-15

    #     alpha = alpha.reshape((*batch_dim, self.K))
    #     mu    = mu.reshape((*batch_dim, self.K, self.model_dim))
    #     sigma = sigma.reshape((*batch_dim, self.K, self.model_dim))
        
    #     mix = dist.Categorical(logits=alpha)
    #     comp = dist.Normal(mu , sigma).to_event(1)
    #     dit = dist.MixtureSameFamily(mix, comp)
        
    #     normalizing = torch_transforms.AffineTransform(
    #         self.ms_means, self.ms_stds, event_dim=1)
        
    #     dit = dist.TransformedDistribution(dit, normalizing)
        
    #     return alpha, mu, sigma, dit
        
    # def log_prob(self, model_samples, data_samples):
        
    #     _, _, _, dit = self.evaluate(data_samples)           
        
    #     # print('dit.batch_shape', dit.batch_shape)
    #     # print('dit.event_shape', dit.event_shape)
        
    #     # print('model_sample', model_sample.shape)
    #     # print('model_sample.expand', model_sample.expand(dit.batch_shape).shape)
        
    #     # print(data_samples.shape)
    #     # print(model_samples.unsqueeze(-1).shape)
    #     # print(model_samples.expand(data_samples.shape[:-1]).unsqueeze(-1).shape)
        
    #     return dit.log_prob(model_samples.unsqueeze(1))

    # def clear_cache(self):
    #     pass
    
    # def sample(self, data_samples):
    #     #TODO make sampling a bit more transparent with how many samples should be drawn from each posterior individually

    #     _, _, _, dit = self.evaluate(data_samples)
                
    #     sample = dit.sample([1,]).flatten(start_dim=0, end_dim=1)
    #     return sample
    
    # def get_mix(self, data_samples):
    #     alpha, _, _, _ = self.evaluate(data_samples)
    #     return alpha
    
    # def get_mean(self, data_samples):
    #     _, mu, _, _ = self.evaluate(data_samples)
    #     return mu
    
    # def get_sigma(self, data_samples):
    #     _, _, sigma, _ = self.evaluate(data_samples)
    #     return sigma


class Cond_Spline_NF(torch.nn.Module):
    
    def __init__(self, data_samples, model_sample):
        super(Cond_Spline_NF, self).__init__()
        
        self.design_dim = data_samples.shape[-1]
        self.model_dim  = model_sample.shape[-1]
        
        self.ds_means = data_samples.mean((0, 1))
        self.ds_stds  = data_samples.std((0, 1))
        
        self.ms_means = model_sample.mean(dim=0)
        self.ms_stds  = model_sample.std(dim=0)
        
        dist_base_m = dist.Normal(torch.zeros(self.model_dim),
                                  torch.ones(self.model_dim))
        
        self.x2_transform_1 = transforms.conditional_spline(
            self.model_dim, context_dim=self.design_dim, count_bins=64)
        self.x2_transform_2 = transforms.conditional_spline(
            self.model_dim, context_dim=self.design_dim, count_bins=64)

        ms_normalizing = torch_transforms.AffineTransform(
            self.ms_means, 10*self.ms_stds, event_dim=1)
        
        self.dist_x2_given_x1 = dist.ConditionalTransformedDistribution(dist_base_m, [self.x2_transform_1, self.x2_transform_2, ms_normalizing])
        
    def log_prob(self, model_space, data_samples):
        
        x = data_samples.clone()
        
        x -= self.ds_means
        x /= self.ds_stds  
        
        dit = self.dist_x2_given_x1.condition(x)       
        
        return dit.log_prob(model_space)

    def parameters(self):
        return torch.nn.ModuleList(
            [self.x2_transform_1, self.x2_transform_2]).parameters()

    def clear_cache(self):
        self.dist_x2_given_x1.clear_cache()
    
    def sample(self, data_samples):
        #TODO make sampling a bit more transparent with how many samples should be drawn from each posterior individually
        
        dit = self.dist_x2_given_x1.condition(data_samples)
                
        sample = dit.sample(data_samples.shape[:-1])
        return sample

