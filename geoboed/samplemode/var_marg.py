import math

import numpy as np
import torch
import pyro

from tqdm import tqdm
import matplotlib.pyplot as plt

import pyro.distributions as dist
from torch.distributions import constraints
import pyro.distributions.transforms as transforms
import torch.distributions.transforms as torch_transforms
from torch.utils.data import Dataset, DataLoader

from .dataloader import DataPrepocessor


def var_marg(self, dataframe, design_list,
             
             var_guide, 

             N, M=-1,
             
             n_batch=1,
             n_epochs=1,
             optim=None, scheduler=None,
             guide_args={},
             preload_samples=False,
             return_guide=False, 
             disable_tqdm=False,
             set_rseed=True,
             plot_loss=None):
    
    # for reproducibility
    if set_rseed:
        pyro.set_rng_seed(0)
        torch.manual_seed(0)
        
    if N > self.n_prior: raise ValueError(
        'Not enough prior samples, N has to be smaller or equal than n_prior!')
        
    # set M if all samples should be used (-1)
    M = M if not M == -1 else self.n_prior - N
    
    # Some checks to make sure all inputs are correct
    if N + M > self.n_prior: raise ValueError(
        'Not enough prior samples, N+M has to be smaller or equal than n_prior!')
    if n_batch > M: raise ValueError(
        'Batch size cant be larger then sample size')
    
    if var_guide == 'nf':
        guide_template = Coupling_Spline_NF
    elif var_guide == 'gmm':
        guide_template = GaussianMixtureModel
    elif var_guide == 'dn':
        guide_template = None
    elif type(var_guide) == str:
        raise NotImplementedError(f'Guide {var_guide} not implemented')
    else:
        guide_template = var_guide

    
    # set optimizer or fall back to reasonable default if None is set
    if optim is None:
        def optimizer_constructor(guide):
            return torch.optim.Adam(guide.parameters(), lr=1e-2)
    else:
        optimizer_constructor = optim
        
    # set scheduler or don't set any if none is set
    if scheduler is not None:
        scheduler_constructor = scheduler
        
    eig_list = []
    if return_guide:
        guide_collection = []
    losses_collection = []
        
    if preload_samples:
        dataframe = dict(dataframe)
        
    for i, design_i in tqdm(enumerate(design_list), total=len(design_list), disable=disable_tqdm):
                
        if set_rseed:
            # set random set so each design point has same noise realisation
            pyro.set_rng_seed(0)
            torch.manual_seed(0)
        
        if var_guide == 'dn':
            # for dn/single gaussian no guide needs to be trained
            N_fwd_samples       = torch.tensor(dataframe['data'][:N, design_i.tolist()])[:, None, :].float()
            N_likeliehood       = self.data_likelihood(N_fwd_samples, self.get_designs()[design_i.tolist()])
            N_likeliehood_samples = N_likeliehood.sample([1, ]).flatten(start_dim=0, end_dim=1).detach()
            N_processed_samples = DataPrepocessor(N_likeliehood_samples.float())
            
            # determinant of 1D array covariance not possible so we need to differentiate
            if len(design_i) < 2:
                model_det = torch.cov(N_processed_samples.data.squeeze()).detach()
            else:
                model_det = torch.linalg.det(torch.cov(N_processed_samples.data.squeeze().T))
            #TODO: Test if this really works                 
            D = design_i.shape[0] 
                        
            marginal_lp = -1/2 * (math.log(model_det) + D/1 + D/1 * math.log(2*math.pi))
            
            conditional_lp = N_likeliehood.log_prob(N_processed_samples.data).detach()
            
            eig = ((conditional_lp).sum(0) / N) - marginal_lp
            
        else:
            # design restriction need to be applied on axis one as axis 0 is the sample dimension 
            # tolist necessary as as single elemet tensor will return an int instead of a tensor
            # None necessary to make make sure the tensor is in the right shape
            N_fwd_samples         = torch.tensor(dataframe['data'][:N, design_i.tolist()])[:, None, :].float()
            M_fwd_samples         = torch.tensor(dataframe['data'][N:N+M, design_i.tolist()])[:, None, :].float()
            # generate noisy samples for each fwd model sample
            
            N_likeliehood = self.data_likelihood(N_fwd_samples, self.get_designs()[design_i.tolist()])
            M_likeliehood = self.data_likelihood(M_fwd_samples, self.get_designs()[design_i.tolist()])
            
            N_likeliehood_samples = N_likeliehood.sample([1, ]).flatten(start_dim=0, end_dim=1).detach()
            M_likeliehood_samples = M_likeliehood.sample([1, ]).flatten(start_dim=0, end_dim=1).detach()
            
            N_processed_samples = DataPrepocessor(N_likeliehood_samples.float())
            M_processed_samples = DataPrepocessor(M_likeliehood_samples.float())
            
            sample_loader = DataLoader(M_processed_samples, batch_size=n_batch, shuffle=True)
                        
            # use all data here to get proper mean and boundaries
            guide = guide_template(torch.cat((N_likeliehood_samples, M_likeliehood_samples), 0), guide_args)
            optimizer = optimizer_constructor(guide)
            if scheduler is not None:
                scheduler = scheduler_constructor(optimizer)
                
            losses = []

            for e in range(n_epochs):
                batch_losses = []
                for ix, samples in enumerate(sample_loader):
                                    
                    loss = -guide.log_prob(samples.detach()).mean()
                    
                    # set_to_none slightly faster than zero_grad
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    guide.clear_cache()

                    losses.append(loss.detach().item())
                    
                    del loss
                        
            marginal_lp = guide.log_prob(N_processed_samples.data).detach()
            guide.clear_cache()
            
            losses_collection.append(losses)

            if return_guide:
                guide_collection.append(guide)
            del guide

            conditional_lp = N_likeliehood.log_prob(N_processed_samples.data).detach()
            eig = (conditional_lp - marginal_lp).sum(0) / N 
                
        eig_list.append(eig.detach().item())
                                        
    if return_guide:
        output_dict = {
            'var_guide_name':  var_guide,
            'var_guide':       guide_collection,
            'N':               N,
            'M':               M,
            'optim':           optim,
            'scheduler':       scheduler,
            'guide_args':      guide_args,
            'losses':          np.array(losses_collection).T,
            }
                        
        return np.array(eig_list), output_dict
    
    elif var_guide == 'dn':
        output_dict = {'N': N,}
        return np.array(eig_list), output_dict
    
    else:
        output_dict = {
            'var_guide_name':  var_guide,
            'N':               N,
            'M':               M,
            'optim':           optim,
            'scheduler':       scheduler,
            'guide_args':      guide_args,
            'losses':          np.array(losses_collection).T,
            }
        return np.array(eig_list), output_dict
        


class Coupling_Spline_NF(torch.nn.Module):

    def __init__(self, data, guide_args={}):
        super().__init__()

        data = data.float()
        self.desing_dim = data.shape[-1]
        
        self.ds_means = data.mean(dim=0)
        self.ds_stds  = data.std(dim=0) 
                
        # if dim is prvided tuple is returned
        self.ds_min = torch.min(data, dim=0)[0]
        self.ds_max = torch.max(data, dim=0)[0]
        self.ds_range = self.ds_max - self.ds_min
        # add small buffer to avoide numerical issues
        self.ds_min -= 1e-1 * self.ds_range
        self.ds_max += 1e-1 * self.ds_range
        
        self.cpl_base_dist = dist.Normal(torch.zeros(self.desing_dim),
                                         torch.ones(self.desing_dim))

        self.transform_list = []
        
        if 'flows' in guide_args: 
            for i_f, flow in enumerate(guide_args['flows']):
                if 'flow_args' in guide_args:            
                    self.transform_list.append(flow(self.desing_dim, **guide_args['flow_args'][i_f]))
                else:
                    self.transform_list.append(flow(self.desing_dim))
        else:
           self.transform_list.append(transforms.spline_coupling(self.desing_dim, count_bins=8))

        self.cpl_trans_dist = dist.TransformedDistribution(
            self.cpl_base_dist, self.transform_list)
            
        if 'processing' in guide_args:

            if guide_args['processing'] == 'normalise':
                # event_dim necessary for some transformations eg spline        
                self.guide  = dist.TransformedDistribution(
                self.cpl_trans_dist ,  torch_transforms.AffineTransform(self.ds_means, self.ds_stds, event_dim=1))
            elif guide_args['processing'] == 'log_transform':
                # should have the same effect as the one used in Zhang 2021: Seismic Tomography Using Variational Inference Methods
                self.guide  = dist.TransformedDistribution(
                    self.cpl_trans_dist ,  torch_transforms.SigmoidTransform())
                self.guide  = dist.TransformedDistribution(
                    self.guide ,  torch_transforms.AffineTransform(self.ds_min, (self.ds_max - self.ds_min), event_dim=1))
        
    def parameters(self):
        return torch.nn.ModuleList(
            self.transform_list
            ).parameters()

    def log_prob(self, data_samples):
        x = data_samples.detach().clone()
        # this should be equivalent?!?
        # x = data_samples.float().detach()        
        log_prob = self.guide.log_prob(x)
        return log_prob
    
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
        
        # if dim is prvided tuple is returned
        self.ds_min = torch.min(data, dim=0)[0]
        self.ds_max = torch.max(data, dim=0)[0]
        self.ds_range = self.ds_max - self.ds_min
        # add small buffer to avoide numerical issues
        self.ds_min -= 1e-1 * self.ds_range
        self.ds_max += 1e-1 * self.ds_range

        self.guide_args = guide_args
        
        if 'processing' in guide_args:
            
            mix_param  = torch.ones(self.K)
            
            if self.guide_args['processing'] == 'normalise':
                mean_param = (data[:self.K] - self.ds_means) / self.ds_stds
                mean_param = mean_param[:, 0, :] 
                cov_param  = torch.diag_embed(torch.ones(self.K, self.data_dim))

            else:
                mean_param = torch.randn(self.K, self.data_dim)
                cov_param  = torch.diag_embed(torch.ones(self.K, self.data_dim))
                        
        else:
            mix_param  = torch.ones(self.K)
            if self.K > 1:
                mean_param = data[:self.K][:, 0, :] # = torch.randn(self.K, self.data_dim)
            else:
                mean_param = self.ds_means # = torch.randn(self.K, self.data_dim)

            cov_param  = torch.diag_embed(self.ds_stds/self.K * torch.ones(self.K, self.data_dim))
            
        self.mix_param  = torch.nn.Parameter(mix_param)
        self.mean_param = torch.nn.Parameter(mean_param)
        self.cov_param  = torch.nn.Parameter(cov_param)
        
    def log_prob(self, data_samples):
        
        # DONT remove the .clone() here. It will break the code because of the way pytorch handles gradients
        # x = data_samples.detach().clone()

        # normalize mixture weights to ensure they sum to 1 (simplex constraint)
        self.mix = dist.Categorical(logits=torch.nn.functional.log_softmax(self.mix_param, -1))
        
        # ensure positive-definiteness
        cov = torch.matmul(self.cov_param, torch.transpose(self.cov_param, -1, -2)) 
        cov += torch.diag_embed(torch.ones(self.K, self.data_dim)) * 1e-5
        
        self.comp = dist.MultivariateNormal(self.mean_param, covariance_matrix=cov)
        self.guide = dist.MixtureSameFamily(self.mix, self.comp)
        
        if 'processing' in self.guide_args:

            if self.guide_args['processing'] == 'normalise':
                # event_dim necessary for some transformations eg spline        
                self.guide  = dist.TransformedDistribution(
                self.guide ,  torch_transforms.AffineTransform(self.ds_means, self.ds_stds, event_dim=1))
            elif self.guide_args['processing'] == 'log_transform':
                # should have the same effect as the one used in Zhang 2021: Seismic Tomography Using Variational Inference Methods
                self.guide  = dist.TransformedDistribution(
                    self.guide ,  torch_transforms.SigmoidTransform())
                self.guide  = dist.TransformedDistribution(
                    self.guide ,  torch_transforms.AffineTransform(self.ds_min, (self.ds_max - self.ds_min), event_dim=1))
            elif self.guide_args['processing'] == 'max_divide': 
                self.guide  = dist.TransformedDistribution(
                    self.guide ,  torch_transforms.AffineTransform(0, self.ds_max, event_dim=1))

        return self.guide.log_prob(data_samples.detach())
    
    def clear_cache(self):
        pass
        
    def sample(self, size):
        sample = self.guide.sample(size)
        return sample