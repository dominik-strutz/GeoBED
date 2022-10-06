import random

import numpy as np
import torch
import pyro

from tqdm import tqdm
import matplotlib.pyplot as plt

import pyro.distributions as dist
from torch.distributions import constraints
import pyro.distributions.transforms as transforms
import torch.distributions.transforms as torch_transforms

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture


def var_marg(self, dataframe, design_list, var_guide,
             n_steps, N, M=-1, n_stochastic=None,
             optim=None, scheduler=None,
             batched=False, guide_args={},
             return_dict=False, preload_samples=True,
             disable_tqdm=False, set_rseed=True, **kwargs):


    # Some checks to make sure all inputs are correct
    if N + M > self.n_prior: raise ValueError(
        'Not enough prior samples, N+M has to be smaller than n_prior!')
    
    # set M properly if all samples should be used (-1)
    M = M if not M == -1 else self.n_prior - N
        
    if n_stochastic is  not None:
        if n_stochastic >= N: raise ValueError(
            'No stochastic sampling possible when all samples are used')
        
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
            
    if preload_samples:
        pre_samples = torch.tensor(np.apply_along_axis(self.design_restriction, 1, dataframe['data'][:N+M]))
    
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
                random.seed(0)
            
            if preload_samples:
                # tolist necessary as as single elemet tensor will return an int instead of a tensor
                # None necessary to make make sure the tnesor is in the right shape
                samples = pre_samples[:N+M, design_i.tolist()][:, None, :]            
            else:
                samples = torch.tensor(np.apply_along_axis(self.design_restriction, 1, dataframe['data'][:N+M]))[design_i.tolist()][:, None, :]       
                        
            if var_guide == 'cpl_spline_nf':
                guide = Coupling_Spline_NF(samples[:], guide_args)
            elif var_guide == 'gmm':
                guide = GaussianMixtureModel(samples[:], guide_args)
            elif var_guide == 'gmm_scikit':
                pass
            elif type(var_guide) == str:
                raise NotImplementedError('Guide not implemented')
                        
            if var_guide != 'gmm_scikit':
                
                optimizer = optimizer_constructor(guide)
                if scheduler is not None:
                    scheduler = scheduler_constructor(optimizer)

                losses = []

                for step in (tqdm(range(n_steps), total=n_steps, disable=True, leave=False)):
                    
                    optimizer.zero_grad()

                    if n_stochastic is not None:
                        random_indices = random.sample(range(N), n_stochastic)
                        x = self.data_likelihood(samples[random_indices], self.get_designs()[design_i]).sample([1, ]).flatten(start_dim=0, end_dim=1)
                        
                    else:
                        x = self.data_likelihood(samples[:N], self.get_designs()[design_i]).sample([1, ]).flatten(start_dim=0, end_dim=1)                
                                
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
                                    
                likeliehood = self.data_likelihood(samples[N:N+M], self.get_designs()[design_i])
                samples = likeliehood.sample([1, ]).flatten(start_dim=0, end_dim=1)

                marginal_lp = guide.log_prob(samples)     
                conditional_lp = likeliehood.log_prob(samples)
                guide.clear_cache()

                eig = (conditional_lp - marginal_lp).sum(0) / M 
                
                eig_list.append(eig.detach().item())
                            
                losses_collection.append(losses)
                if return_dict:
                    guide_collection.append(guide)
                    
                del guide
            
            else:
                guide_args['max_iter'] = n_steps
                guide_args['random_state'] = 0
                # print(samples.shape)
                
                ds_means = samples.mean(dim=0)                
                ds_stds  = samples.std(dim=0) 
                                
                # normalize to make convergence possible... #TODOL make preprocessing more general and let all functions use the same
                N_likeliehood = self.data_likelihood((samples[:N] - ds_means) / ds_stds, self.get_designs()[design_i])
                N_samples = N_likeliehood.sample([1, ]).flatten(start_dim=0, end_dim=1)                      
                
#                 if (i == 50) and (len(design_list[0]) == 2):
#                     guide_args['verbose'] = True

#                     guide = GaussianMixture(**guide_args)
#                     guide = guide.fit((N_samples[..., 0,  :]).clone().detach().numpy())
                    
#                     print(N_samples[..., 0,  :].detach().numpy().shape)
#                     # print(guide.get_params())
#                     # print(guide.means_)
#                     # print(guide.weights_)
#                     # print(guide.covariances_)
#                     # print(guide.n_iter_)
                    
#                     test_samples, _ = guide.sample(int(1e3))
#                     plt.scatter(N_samples[:, 0, 0], N_samples[:, 0, 1], marker='x', s=20, label='reference')
#                     plt.legend()
#                     plt.show()
                    
#                     plt.scatter(test_samples[:, 0], test_samples[:, 1], label='fit')
#                     plt.legend()
#                     plt.show()
                    
#                     plt.scatter(N_samples[:, 0, 0], N_samples[:, 0, 1], marker='x', s=20, label='reference')
#                     plt.scatter(test_samples[:, 0], test_samples[:, 1], label='fit')
#                     plt.legend()
#                     plt.show()
                    
#                     guide_args['verbose'] = False

                    
                # else:
                guide = BayesianGaussianMixture(**guide_args).fit(N_samples[..., 0,  :].detach().numpy())

                # print(guide)
                # print(guide.converged_)
                # print(guide.weights_)
                # print(guide.n_iter_)

                M_likeliehood = self.data_likelihood((samples[N:N+M] - ds_means) / ds_stds, self.get_designs()[design_i])
                M_samples = M_likeliehood.sample([1, ]).flatten(start_dim=0, end_dim=1)

                marginal_lp = torch.tensor(guide.score_samples(M_samples[..., 0, :].detach().numpy())).unsqueeze(-1)
                conditional_lp = M_likeliehood.log_prob(M_samples)
                                
                eig = (conditional_lp - marginal_lp).sum(0) / M                
                eig_list.append(eig.detach().item())
                                
                losses_collection.append([guide.lower_bound_, ])          
                
        if return_dict:
            output_dict = {
                'var_guide_name':  var_guide,
                'var_guide':       guide_collection,
                'n_steps':         n_steps,
                'N':               N,
                'M':               M,
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
        
        # DONT remove the .clone() here. It will break the code because of the way pytorch handles gradients
        x = data_samples.clone()

        # normalize mixture weights to ensure they sum to 1 (simplex constraint)
        self.mix = dist.Categorical(logits=torch.nn.functional.log_softmax(self.mix_param, -1))
        
        # ensure positive-definiteness
        cov = torch.matmul(self.cov_param, torch.transpose(self.cov_param, -1, -2)) 
        cov += torch.diag_embed(torch.ones(self.K, self.data_dim)) * 1e-6
        
        self.comp = dist.MultivariateNormal(self.mean_param, covariance_matrix=cov)
        self.guide = dist.MixtureSameFamily(self.mix, self.comp)
        
        # This makes allows the guide to be fitted a lot quicker 
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
