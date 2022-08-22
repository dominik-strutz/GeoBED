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
             **kwargs):

    #TODO: check the influence of using the same samples for training and evaluating
    
    #TODO: sometimes gradient descent fails due to wrong covarianxce matrix... fix that 
    
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
        
    #TODO: Implement way to do eig optimization batched (easily done for gmm, not so much for nf)
    
    if preload_samples:
        pre_samples = dataframe['data'][:n_final_samples]
    
    if batched:
        
        print('loading samples ...')
        if preload_samples:
            samples = np.concatenate([pre_samples[:n_final_samples, design_i][:, None, :] for design_i in design_list], axis=1)            
        else:
            samples = np.concatenate([dataframe['data'][:n_final_samples, design_i][:, None, :] for design_i in design_list], axis=1)            
        samples = torch.tensor(samples)
        print('finished loading samples ...')

        # set random set so each design point has same noise realisation
        # pyro.set_rng_seed(0)
        
        #TODO look for way to make noise consistent when batched
        if var_guide == 'gmm':
            guide = GaussianMixtureModel_batched(samples, **guide_args)
        elif var_guide == 'nf_cpl': 
            guide = Coupling_Spline_NF_batched(samples, **guide_args) 
        elif type(var_guide) == str:
            raise NotImplementedError('Guide not implemented for batched calculations.')
        
        optimizer = optimizer_constructor(guide)
        if scheduler is not None:
            scheduler = scheduler_constructor(optimizer)
        
        losses = []
        
        for step in (tqdm(
            range(n_steps), total=n_steps, disable=self.disable_tqdm, leave=True)):
            
            optimizer.zero_grad()
            
            if stoch_sampling:
                random_indices = random.sample(range(n_samples), n_stochastic)
                x = self.data_likelihood(samples[random_indices]).sample([1, ]).flatten(start_dim=0, end_dim=1)
                
            else:
                x = self.data_likelihood(samples[:n_samples]).sample([1, ]).flatten(start_dim=0, end_dim=1)                             
            
            loss = -guide.log_prob(x).mean(dim=0)
            
            loss.backward(torch.ones_like(loss))
            optimizer.step()
            guide.clear_cache()
        
            if return_dict:
                losses.append(loss.detach().tolist())
            
            # if not self.disable_tqdm:
            #     pbar.set_description(f"avg. Loss: {loss.sum().detach().item():.2e}")

        #TODO: implement option to not reuse samples for final samples  

        likeliehood = self.data_likelihood(samples)
        samples = likeliehood.sample([1, ]).flatten(start_dim=0, end_dim=1)

        marginal_lp = guide.log_prob(samples)     
        conditional_lp = likeliehood.log_prob(samples)
                
        eig = (conditional_lp - marginal_lp).sum(0) / n_final_samples 
        
        eig_list = eig.detach().tolist()
                        
                        
        if return_dict:
            output_dict = {
                'var_guide_name':  var_guide,
                'var_guide'     :  guide,
                'n_steps':         n_steps,
                'n_samples':       n_samples,
                'n_final_samples': n_final_samples,
                'stoch_sampling':  stoch_sampling,
                'n_stochastic':    n_stochastic,
                'optim':           optim,
                'scheduler':       scheduler,
                'batched':         batched,
                'guide_args':      guide_args,
                'losses':          np.array(losses),
                }
            return np.array(eig_list), output_dict
        else:
            return np.array(eig_list), None
        
    else:
        # safe output for each design step to those lists #TODO make this more sophisticated 
        
        eig_list = []
        if return_dict:
            guide_collection = []
        losses_collection = []
        
        
        for i, design_i in tqdm(enumerate(design_list), total=len(design_list), disable=self.disable_tqdm):
            
            # set random set so each design point has same noise realisation
            pyro.set_rng_seed(0)

            samples = torch.tensor(dataframe['data'][ :n_final_samples, design_i])
                    
            if var_guide == 'spline_nf':
                guide = Spline_NF(samples, **guide_args)
            elif var_guide == 'cpl_spline_nf':
                guide = Coupling_Spline_NF(samples, **guide_args)
            elif type(var_guide) == str:
                raise NotImplementedError('Guide not implemented')
            
            optimizer = optimizer_constructor(guide)
            if scheduler is not None:
                scheduler = scheduler_constructor(optimizer)

            losses = []

            for step in (pbar := tqdm(range(n_steps), total=n_steps, disable=True, leave=False)):
                
                optimizer.zero_grad()

                if stoch_sampling:
                    random_indices = random.sample(range(n_samples), n_stochastic)
                    x = self.data_likelihood(samples[random_indices]).sample([1, ]).flatten(start_dim=0, end_dim=1)
                    
                else:
                    x = self.data_likelihood(samples[:n_samples]).sample([1, ]).flatten(start_dim=0, end_dim=1)                
                            
                loss = -guide.log_prob(x).mean()
                loss.backward()
                optimizer.step()
                guide.clear_cache()
                
                if scheduler is not None:
                    if step % scheduler_n_iterations == 0:
                        scheduler.step()
                
                losses.append(loss.detach().item())
                pbar.set_description(f"Loss: {loss.detach().item():.2e}")
            
                #TODO: some sort of quality control for convergence
                                
            # TODO: Implement way to not reuse samples
            likeliehood = self.data_likelihood(samples, design_i)
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
         

class Spline_NF(torch.nn.Module):

    def __init__(self, data):
        super().__init__()        
       
        self.ds_means = data.mean(dim=0)
        self.ds_stds  = data.std(dim=0)   
        
        self.desing_dim = data.shape[-1]
                
        self.base_dist = dist.Normal(torch.zeros(self.desing_dim),
                                     torch.ones(self.desing_dim))
        self.spline_transform = transforms.Spline(self.desing_dim, count_bins=32)
        self.guide = dist.TransformedDistribution(self.base_dist, [self.spline_transform])

        normalizing = torch_transforms.AffineTransform(self.ds_means, 10*self.ds_stds, event_dim=1)
        self.guide  = dist.TransformedDistribution(self.guide , normalizing)

    def parameters(self):
        return self.spline_transform.parameters()

    def log_prob(self, data_samples):
        x = data_samples.clone()
        return self.guide.log_prob(x)
    
    def clear_cache(self):
        self.guide.clear_cache()
        
    def sample(self, size):
        sample = self.guide.sample(size)
        return sample


class Coupling_Spline_NF(torch.nn.Module):

    def __init__(self, data):
        super().__init__()
                
        self.desing_dim = data.shape[-1]
        
        self.ds_means = data.mean(dim=0)
        self.ds_stds  = data.std(dim=0) 
        
        self.cpl_base_dist = dist.Normal(torch.zeros(self.desing_dim),
                                         torch.ones(self.desing_dim))
        
        self.cpl_spline_transform_1 = transforms.spline_coupling(self.desing_dim, count_bins=16)
        self.cpl_spline_transform_2 = transforms.spline_coupling(self.desing_dim, count_bins=16)

        self.guide = dist.TransformedDistribution(
            self.cpl_base_dist, [self.cpl_spline_transform_1, self.cpl_spline_transform_2])

        normalizing = torch_transforms.AffineTransform(self.ds_means, 10*self.ds_stds)
        self.guide  = dist.TransformedDistribution(self.guide , normalizing)

    def parameters(self):
        return torch.nn.ModuleList(
            [self.cpl_spline_transform_1, self.cpl_spline_transform_2]
            ).parameters()

    def log_prob(self, data_samples):
        x = data_samples.clone()
        return self.guide.log_prob(x)
    
    def clear_cache(self):
        self.guide.clear_cache()
        
    def sample(self, size):
        sample = self.guide.sample(size)
        return sample



class Coupling_Spline_NF_batched(torch.nn.Module):
    def __init__(self, data):
        super().__init__()
        
        self.data_dim = data.shape[-1]
        self.desing_dim = data.shape[-2]
        
        self.ds_means = data.mean(dim=0)
        self.ds_stds  = data.std(dim=0)
        
        print(data.shape)

        self.flows = torch.nn.ModuleList(
            [Coupling_Spline_NF(data=data[:, d_i]) for d_i in range(self.desing_dim)]
            )        
        
    def log_prob(self, data_samples):
        
        x = data_samples.clone()
        # print(self.flows[:2])

        log_probs = torch.empty((x.shape[0], self.desing_dim))
        for i, l in enumerate(self.flows):
            # print(l.log_prob(x).shape)
            log_probs[:, i] = l.log_prob(x[:, i])
        
        return log_probs
      
    def clear_cache(self):
        pass
    
    def sample(self, size):
        sample = self.guide.sample(size)
        return sample

    
class GaussianMixtureModel_batched(torch.nn.Module):

    # TODO: it might be necessary to reverse order of prior samples and designs to make this work

    def __init__(self, data, K=2):
        super().__init__()
        
        # print('data.shape', data.shape)
        
        self.K = K
        self.data_dim = data.shape[-1]
        self.desing_dim = data.shape[-2]
        
        self.ds_means = data.mean(dim=0)
        self.ds_stds  = data.std(dim=0)
        
        # print(self.ds_means.shape)
        # print(self.ds_stds.shape)  

        mix_param  = torch.rand(self.desing_dim, self.K)
        mean_param = torch.rand(self.desing_dim, self.K, self.data_dim)
        cov_param  = torch.diag_embed(torch.ones(self.desing_dim, self.K, self.data_dim))
        
        # print('mix_param.shape', mix_param.shape)
        # print('mean_param.shape', mean_param.shape)
        # print('cov_param.shape', cov_param.shape)
        
        self.mix_param  = torch.nn.Parameter(mix_param)
        self.mean_param = torch.nn.Parameter(mean_param)
        self.cov_param  = torch.nn.Parameter(cov_param)
    
    def log_prob(self, data_samples):
                
        x = data_samples.clone()

        self.mix = dist.Categorical(logits=self.mix_param)
        
        # ensure positive-definiteness (bmm because batch of matrix is used)
        cov = torch.matmul(self.cov_param, torch.transpose(self.cov_param, -1, -2)) + \
            torch.diag_embed(torch.ones(self.desing_dim, self.K, self.data_dim)) * 1e-5
                
        self.comp = dist.MultivariateNormal(self.mean_param, covariance_matrix=cov)
        
        self.guide = dist.MixtureSameFamily(self.mix, self.comp)
        # print('means', self.ds_means.shape)
        # print('std', self.ds_stds.shape)
        
        # print('self.guide.event_shape', self.guide.event_shape)
        # print('self.guide.batch_shape', self.guide.batch_shape)
                
        normalizing = torch_transforms.AffineTransform(self.ds_means, self.ds_stds, event_dim=1)
        
        self.guide  = dist.TransformedDistribution(self.guide , normalizing)     

        return self.guide.log_prob(x)
    
    def clear_cache(self):
        pass
        
    def sample(self, size):
        sample = self.guide.sample(size)
        return sample
