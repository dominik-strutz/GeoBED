import random

import numpy as np
import torch
import pyro

from tqdm import tqdm

import pyro.distributions as dist
from torch.distributions import constraints
import pyro.distributions.transforms as transforms
import torch.distributions.transforms as torch_transforms


def var_marg(self, dataframe, design_list, var_guide,
             n_steps, n_samples, n_final_samples=-1,
             stoch_sampling=True, n_stochastic=None,
             optim=None, scheduler=None,
             batched=False, guide_args={},
             return_dict=False, preload_samples=True,
             disable_tqdm=False, set_rseed=True, **kwargs):

    # for reproducibility
    if set_rseed:
        pyro.set_rng_seed(0)
        torch.manual_seed(0)

    #TODO: check the influence of using the same samples for training and evaluating
    #TODO: sometimes gradient descent fails due to wrong covarianxce matrix... fix that 
    
    # Some checks to make sure all inputs are correct
    n_samples       = n_samples       if not n_samples       == -1 else self.n_prior
    n_final_samples = n_final_samples if not n_final_samples == -1 else self.n_prior
    
    if n_samples  > self.n_prior: raise ValueError(
        'Not enough prior samples, choose different n_samples!')
    if n_final_samples > self.n_prior: raise ValueError(
        'Not enough prior samples, choose different n_final_samples!')
    if stoch_sampling:
        if n_stochastic >= n_samples: raise ValueError(
            'No stochastic sampling possible when all samples are used')
    
    n_max = max(n_samples, n_final_samples)
    
    # set optimizer or fall back to default if none is set
    if optim is None:
        def optimizer_constructor(guide):
            return torch.optim.Adam(guide.parameters(), lr=1e-2)
    else:
        optimizer_constructor = optim
        
    # set scheduler or don't set any if none is set
    if scheduler is not None:
        scheduler_constructor  = scheduler['scheduler']
        scheduler_n_iterations = scheduler['n_iterations']
        
    #TODO: Implement way to do eig optimization batched (!!NOT easily done for gmm)
    
    if preload_samples:
        pre_samples = dataframe['data'][:n_max]
    
    if batched:
        
        raise NotImplementedError('Batched calculations not implemented yet.')
        
    else:        
        eig_list = []
        if return_dict:
            guide_collection = []
        losses_collection = []
        
        
        for i, design_i in tqdm(enumerate(design_list), total=len(design_list), disable=disable_tqdm):
            
            if set_rseed:
                # set random set so each design point has same noise realisation
                pyro.set_rng_seed(0)
                torch.manual_seed(0)
            
            if preload_samples:
                # tolist necessary as as single elemet tensor will return an int instead of a tensor
                samples = pre_samples[:n_max, design_i.tolist()][:, None, :]            
            else:
                samples = dataframe['data'][:n_max, design_i.tolist()][:, None, :]       
            
            samples = torch.tensor(samples)
            
            if var_guide == 'cpl_spline_nf':
                guide = Coupling_Spline_NF(samples, guide_args)
            elif var_guide == 'gmm':
                guide = GaussianMixtureModel(samples, guide_args)
            elif type(var_guide) == str:
                raise NotImplementedError('Guide not implemented')
            
            optimizer = optimizer_constructor(guide)
            if scheduler is not None:
                scheduler = scheduler_constructor(optimizer)

            losses = []

            for step in (tqdm(range(n_steps), total=n_steps, disable=True, leave=False)):
                
                optimizer.zero_grad()

                if stoch_sampling:
                    random_indices = random.sample(range(n_samples), n_stochastic)
                    x = self.data_likelihood(samples[random_indices], self.get_designs()[design_i]).sample([1, ]).flatten(start_dim=0, end_dim=1)
                    
                else:
                    x = self.data_likelihood(samples[:n_samples], self.get_designs()[design_i]).sample([1, ]).flatten(start_dim=0, end_dim=1)                
                            
                loss = -guide.log_prob(x).mean()
                loss.backward()
                optimizer.step()
                guide.clear_cache()
                
                if scheduler is not None:
                    if step % scheduler_n_iterations == 0:
                        scheduler.step()
                
                losses.append(loss.detach().item())
                # pbar.set_description(f"Loss: {loss.detach().item():.2e}")
            
                #TODO: some sort of quality control for convergence
                                
            # TODO: Implement way to not reuse samples
            likeliehood = self.data_likelihood(samples[:n_final_samples], self.get_designs()[design_i])
            samples = likeliehood.sample([1, ]).flatten(start_dim=0, end_dim=1)

            marginal_lp = guide.log_prob(samples)     
            conditional_lp = likeliehood.log_prob(samples)
                         
            eig = (conditional_lp - marginal_lp).sum(0) / n_final_samples 
            
            eig_list.append(eig.detach().item())
                         
            losses_collection.append(losses)
            if return_dict:
                guide_collection.append(guide)  
                
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
         


class Coupling_Spline_NF(torch.nn.Module):

    def __init__(self, data, guide_args={}):
        super().__init__()

        data = data.float()
        self.desing_dim = data.shape[-1]
        
        self.ds_means = data.mean(dim=0)
        self.ds_stds  = data.std(dim=0) 
        
        self.cpl_base_dist = dist.Normal(torch.zeros(self.desing_dim),
                                         torch.ones(self.desing_dim))
        
        self.transform_list = []
        
        if 'flows' in guide_args: 
            for i_f, flow in enumerate(guide_args['flows']):
                self.transform_list.append(flow(self.desing_dim, **guide_args['flow_args'][i_f]))
        else:
           self.transform_list.append(transforms.spline_coupling(self.desing_dim, count_bins=32))

        self.guide = dist.TransformedDistribution(
            self.cpl_base_dist, self.transform_list)

        normalizing = torch_transforms.AffineTransform(self.ds_means, 3*self.ds_stds)
        
        self.guide  = dist.TransformedDistribution(self.guide , normalizing)

    def parameters(self):
        return torch.nn.ModuleList(
            self.transform_list
            ).parameters()

    def log_prob(self, data_samples):
        x = data_samples.float().clone()
        return self.guide.log_prob(x)
    
    def clear_cache(self):
        self.guide.clear_cache()
        
    def sample(self, size):
        sample = self.guide.sample(size)
        return sample


class GaussianMixtureModel(torch.nn.Module):

    # TODO: it might be necessary to reverse order of prior samples and designs to make this work

    def __init__(self, data, guide_args={}):
        super().__init__()
                
        self.K = guide_args['K'] if 'K' in guide_args else 2
        self.data_dim = data.shape[-1]
        
        self.ds_means = data.mean(dim=0)
        self.ds_stds  = data.std(dim=0)

        mix_param  = torch.rand(self.K)
        mean_param = torch.rand(self.K, self.data_dim)
        cov_param  = torch.diag_embed(torch.ones(self.K, self.data_dim))
        
        self.mix_param  = torch.nn.Parameter(mix_param)
        self.mean_param = torch.nn.Parameter(mean_param)
        self.cov_param  = torch.nn.Parameter(cov_param)
    
    def log_prob(self, data_samples):
        
        x = data_samples.clone()

        self.mix = dist.Categorical(logits=self.mix_param)
        
        # ensure positive-definiteness (bmm because batch of matrix is used)
        cov = torch.matmul(self.cov_param, torch.transpose(self.cov_param, -1, -2)) + \
            torch.diag_embed(torch.ones(self.K, self.data_dim)) * 1e-5        
        
        self.comp = dist.MultivariateNormal(self.mean_param, covariance_matrix=cov)
        self.guide = dist.MixtureSameFamily(self.mix, self.comp)
                
        normalizing = torch_transforms.AffineTransform(self.ds_means, self.ds_stds, event_dim=1)
        self.guide  = dist.TransformedDistribution(self.guide , normalizing)     

        return self.guide.log_prob(x)
    
    def clear_cache(self):
        pass
        
    def sample(self, size):
        sample = self.guide.sample(size)
        return sample

    
# class GaussianMixtureModel_batched(torch.nn.Module):

#     # TODO: This is not working yet a as batching introduces information flwo between desing points
#     # it might be necessart to take the same approach as with var post and set design as nn input somehow
    
#     def __init__(self, data, K=2):
#         super().__init__()
        
#         # print('data.shape', data.shape)
        
#         self.K = K
#         self.data_dim = data.shape[-1]
#         self.desing_dim = data.shape[-2]
        
#         self.ds_means = data.mean(dim=0)
#         self.ds_stds  = data.std(dim=0)
        
#         # print(self.ds_means.shape)
#         # print(self.ds_stds.shape)  

#         mix_param  = torch.rand(self.desing_dim, self.K)
#         mean_param = torch.rand(self.desing_dim, self.K, self.data_dim)
#         cov_param  = torch.diag_embed(torch.ones(self.desing_dim, self.K, self.data_dim))
        
#         # print('mix_param.shape', mix_param.shape)
#         # print('mean_param.shape', mean_param.shape)
#         # print('cov_param.shape', cov_param.shape)
        
#         self.mix_param  = torch.nn.Parameter(mix_param)
#         self.mean_param = torch.nn.Parameter(mean_param)
#         self.cov_param  = torch.nn.Parameter(cov_param)
    
#     def log_prob(self, data_samples):
                
#         x = data_samples.clone()

#         self.mix = dist.Categorical(logits=self.mix_param)
        
#         # ensure positive-definiteness (bmm because batch of matrix is used)
#         cov = torch.matmul(self.cov_param, torch.transpose(self.cov_param, -1, -2)) + \
#             torch.diag_embed(torch.ones(self.desing_dim, self.K, self.data_dim)) * 1e-8
                
#         self.comp = dist.MultivariateNormal(self.mean_param, covariance_matrix=cov)
        
#         self.guide = dist.MixtureSameFamily(self.mix, self.comp)
#         # print('means', self.ds_means.shape)
#         # print('std', self.ds_stds.shape)
        
#         # print('self.guide.event_shape', self.guide.event_shape)
#         # print('self.guide.batch_shape', self.guide.batch_shape)
                
#         normalizing = torch_transforms.AffineTransform(self.ds_means, self.ds_stds, event_dim=1)
        
#         self.guide  = dist.TransformedDistribution(self.guide , normalizing)     

#         return self.guide.log_prob(x)
    
#     def clear_cache(self):
#         pass
        
#     def sample(self, size):
#         sample = self.guide.sample(size)
#         return sample
