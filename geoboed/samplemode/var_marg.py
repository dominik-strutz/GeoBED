
import numpy as np
import torch
import pyro
from tqdm import tqdm
import random

import matplotlib.pyplot as plt

import pyro.distributions as dist
from torch.distributions import constraints
import pyro.distributions.transforms as transforms
import torch.distributions.transforms as torch_transforms

import pyro


def var_marg(self, hdf, var_guide, n_steps,
             n_samples, n_final_samples=-1,
             stoch_sampling=True, n_stochastic=None,
             optim=None, scheduler=None,
             n_load=-1, batched=False, guide_args={},
             plot_loss=True, return_guide=False,
             design_selection=None,
             **kwargs):

    
    # TODO: check the influence of using the same samples for training and evaluating
    
    n_samples       = n_samples       if not n_samples       == -1 else self.n_prior
    n_final_samples = n_final_samples if not n_final_samples == -1 else self.n_prior
    
    if n_samples  > self.n_prior: raise ValueError(
        'Not enough prior samples, choose different n_samples!')
    if n_final_samples > self.n_prior: raise ValueError(
        'Not enough prior samples, choose different n_final_samples!')
    
    if stoch_sampling:
        if n_stochastic >= n_samples: raise ValueError(
            'No stochastic sampling possible when all samples are used')
        
    # safe output for each design step to those lists #TODO make this more sophisticated 
    eig_list = []
    if return_guide:
        var_list = []
        
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
    if batched:
        
        print('loading data....')
        samples = torch.tensor(hdf['data'][ :n_samples, :])
        print('batched !!!!')
        
        # #TODO look for way to make noise consistent when batched
        # if var_guide == 'gmm':
        #     guide = GaussianMixtureModel_batched(samples[: , :n_samples], **guide_args)
        # elif type(var_guide) == str:
        #     raise NotImplementedError('Guide not implemented')
        
        # optimizer = optimizer_constructor(guide)
        
        # losses = []
        
        # for step in (pbar := tqdm(
        #     range(n_steps), total=n_steps, disable=self.disable_tqdm, leave=True)):
                
        #     if stoch_sampling:                    
        #         random_indices = random.sample(range(N_max), n_samples)
                
        #         x = self.data_likelihood(samples[:, random_indices]).sample([1, ]).reshape(
        #             (self.n_design, n_samples, self.design_dim)
        #         )
                
        #         # print(x.shape)
                
        #     else:
        #         x = self.data_likelihood(samples[:n_samples]).sample([1, ]).flatten(start_dim=0, end_dim=1)                
                        
        #     optimizer.zero_grad()
        #     loss = -guide.log_prob(x).mean()
        #     loss.backward()
        #     optimizer.step()
        #     guide.clear_cache()
        
        #     losses.append(loss.detach().item())
        #     pbar.set_description(f"Loss: {loss.detach().item():.2e}")
            
        # var_list.append(guide)
        
        # var_samples = guide.sample(torch.Size([final_n_samples]))
        
        # marginal_lp = guide.log_prob(var_samples)
        
        # var_likelihoods = self.data_likelihood(var_samples)
        # conditional_lp = var_likelihoods.log_prob(var_samples).squeeze()
        # eig = (conditional_lp - marginal_lp).sum(0) / N_max 
        
        # eig_list = eig.detach()              
            

        
    else:
    
        for i, design_i in tqdm(enumerate(hdf['design']),
                                total=self.n_design,
                                disable=self.disable_tqdm):
            
            if design_selection is not None:
                if i not in design_selection:
                    continue
            
            # set random set so each design point has same noise realisation
            pyro.set_rng_seed(0)

            samples = torch.tensor(hdf['data'][:n_samples, i])         
                    
            if var_guide == 'spline_nf':
                guide = Spline_NF(samples, **guide_args)
            elif var_guide == 'cpl_spline_nf':
                guide = Coupling_Spline_NF(samples, **guide_args)
            elif var_guide == 'gmm':
                guide = GaussianMixtureModel(samples, **guide_args)
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
                    x = self.data_likelihood(samples).sample([1, ]).flatten(start_dim=0, end_dim=1)                
                            
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
            samples = torch.tensor(hdf['data'][:n_final_samples, i])         

            likeliehood = self.data_likelihood(samples, design_i)
            final_samples = likeliehood.sample([1, ]).flatten(start_dim=0, end_dim=1)
        
            marginal_lp = guide.log_prob(final_samples)     
            conditional_lp = likeliehood.log_prob(final_samples)

            eig = (conditional_lp - marginal_lp).sum(0) / n_final_samples 
            
            eig_list.append(eig.detach().item())
                         
            if return_guide:
                var_list.append(guide)
                
            if plot_loss:
                color = plt.cm.viridis(np.linspace(0, 1, self.n_design))
                plt.plot(losses, color=color[i])     
        
    if return_guide:
        return np.array(eig_list), var_list
    else:
        return np.array(eig_list)
    

class VarMarg_Base(torch.nn.Module):
    def __init__(self):
        super().__init__()        

    def log_prob(self, x):
        return self.guide.log_prob(x)
    
    def clear_cache(self):
        self.guide.clear_cache()
        
    def sample(self, size):
        sample = self.guide.sample(size)
        return sample


class Spline_NF(VarMarg_Base):

    def __init__(self, data):
        super().__init__()        
       
        self.ds_means = data.mean(dim=0)
        self.ds_stds  = data.std(dim=0)   
        
        self.desing_dim = data.shape[-1]
                
        self.base_dist = dist.Normal(torch.zeros(self.desing_dim),
                                     torch.ones(self.desing_dim))
        self.spline_transform = transforms.Spline(self.desing_dim, count_bins=16)
        self.guide = dist.TransformedDistribution(self.base_dist, [self.spline_transform])

        normalizing = torch_transforms.AffineTransform(self.ds_means, self.ds_stds, event_dim=1)
        self.guide  = dist.TransformedDistribution(self.guide , normalizing)

    def parameters(self):
        return self.spline_transform.parameters()


class Coupling_Spline_NF(VarMarg_Base):

    def __init__(self, data):
        super().__init__()
        
        self.desing_dim = data.shape[-1]
        
        self.ds_means = data.mean(dim=0)
        self.ds_stds  = data.std(dim=0) 
        
        self.cpl_base_dist = dist.Normal(torch.zeros(self.desing_dim),
                                         torch.ones(self.desing_dim))
        
        self.cpl_spline_transform_1 = transforms.spline_coupling(self.desing_dim, count_bins=16)

        self.guide = dist.TransformedDistribution(
            self.cpl_base_dist, [self.cpl_spline_transform_1])

        normalizing = torch_transforms.AffineTransform(self.ds_means, self.ds_stds, event_dim=1)
        self.guide  = dist.TransformedDistribution(self.guide , normalizing)

    def parameters(self):
        return torch.nn.ModuleList(
            [self.cpl_spline_transform_1]
            ).parameters()


class GaussianMixtureModel(VarMarg_Base):

    def __init__(self, data, K=2):
        super().__init__()
        
        self.K = K
        self.desing_dim = data.shape[-1]
        
        self.ds_means = data.mean(dim=0)
        self.ds_stds  = data.std(dim=0)   

        mix_param  = torch.rand(self.K)
        mean_param = torch.rand(self.K, self.desing_dim)
        cov_param  = torch.diag_embed(torch.ones(self.K, self.desing_dim))
        
        self.mix_param = torch.nn.Parameter(mix_param)
        self.mean_param   = torch.nn.Parameter(mean_param)
        self.cov_param  = torch.nn.Parameter(cov_param)
    
    def log_prob(self, x):

        self.mix = dist.Categorical(logits=self.mix_param)
        
        # ensure positive-definiteness
        cov = torch.matmul(self.cov_param, torch.transpose(self.cov_param, -1, -2))
        self.comp = dist.MultivariateNormal(self.mean_param, covariance_matrix=cov)
        
        self.guide = dist.MixtureSameFamily(self.mix, self.comp)
        
        normalizing = torch_transforms.AffineTransform(self.ds_means, self.ds_stds, event_dim=1)
        self.guide  = dist.TransformedDistribution(self.guide , normalizing)  
        
        return self.guide.log_prob(x)
    
    def clear_cache(self):
        pass
    
    
class GaussianMixtureModel_batched(VarMarg_Base):

    # TODO: it might be necessary to reverse order of prior samples and designs to make this work

    def __init__(self, data, K=2):
        super().__init__()
        
        # print('data.shape', data.shape)
        
        self.K = K
        self.desing_dim = data.shape[-1]
        self.batch_dim = data.shape[1]
        
        self.ds_means = data.mean(dim=-2, keepdim=True)
        self.ds_stds  = data.std(dim=-2, keepdim=True)   

        mix_param  = torch.rand(self.batch_dim, self.K)
        mean_param = torch.rand(self.batch_dim, self.K, self.desing_dim)
        cov_param  = torch.diag_embed(torch.ones(self.batch_dim, self.K, self.desing_dim))

        # print(mix_param.shape)
        # print(mean_param.shape)
        # print(cov_param.shape)
        
        self.mix_param = torch.nn.Parameter(mix_param)
        self.mean_param   = torch.nn.Parameter(mean_param)
        self.cov_param  = torch.nn.Parameter(cov_param)
    
    def log_prob(self, x):

        self.mix = dist.Categorical(logits=self.mix_param)
        
        # ensure positive-definiteness (bmm because batch of matrix is used)
        cov = torch.matmul(self.cov_param, torch.transpose(self.cov_param, -1, -2))
                
        self.comp = dist.MultivariateNormal(self.mean_param, covariance_matrix=cov)
        
        self.guide = dist.MixtureSameFamily(self.mix, self.comp)
        
        # print('means', self.ds_means.shape)
        # print('std', self.ds_stds.shape)
        
        # print('self.guide.batch_shape', self.guide.batch_shape)
        
        normalizing = torch_transforms.AffineTransform(self.ds_means, self.ds_stds, event_dim=2)
        self.guide  = dist.TransformedDistribution(self.guide , normalizing)     
        
        # print(x.shape)
        
        # print('self.guide.batch_shape', self.guide.batch_shape)
        
        return self.guide.log_prob(x)
    
    def clear_cache(self):
        pass