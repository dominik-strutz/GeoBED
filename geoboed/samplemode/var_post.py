import random

import torch
from torch import nn

import numpy as np
import pyro
from tqdm import tqdm

import pyro.distributions as dist
import pyro.distributions.transforms as transforms
import torch.distributions.transforms as torch_transforms
from torch.utils.data import Dataset, DataLoader

from .dataloader import DataPrepocessor


def var_post(self, dataframe, design_list,
             
             var_guide,
             
             N, M=-1,
             
             n_batch=1,
             n_epochs=1,
             optim=None, scheduler=None,
             
             guide_args={},
             
             model_prior=None,

             return_guide=False,
             interrogation_mapping=None,
             
             disable_tqdm=False,
             set_rseed=True,
             plot_loss=None):

    # N = number of samples for the variational training
    # M = number of samples for the Monte Carlo approximation
    
    # for reproducibility
    if set_rseed:
        pyro.set_rng_seed(0)
        torch.manual_seed(0)

    # Some checks to make sure all inputs are correct
    if N + M > self.n_prior: raise ValueError(
        'Not enough prior samples, N+M has to be smaller than n_prior!')
    if n_batch >= M: raise ValueError(
        'No stochastic sampling possible when all samples are used')
    
    # set M properly if all samples should be used (-1)
    M = M if not M == -1 else self.n_prior - N
        
    # set costum optimizer or fall back to default if none is set
    if optim is None:
        def optimizer_constructor(guide):
            return torch.optim.Adam(guide.parameters(), lr=1e-2)
    else:
        optimizer_constructor = optim
        
    # set costum scheduler or fall back to default if none is set
    if scheduler is not None:
        scheduler_constructor  = scheduler
    
    if var_guide == 'mdn':
        guide_template = MDN_guide
    elif var_guide == 'mdn_cov':
        guide_template = MDN_guide_cov
    elif var_guide == 'cond_nf':
        guide_template = Cond_Spline_NF
    elif type(var_guide) == str:
        raise NotImplementedError(f'Guide {var_guide} not implemented')
    else:
        guide_template = var_guide

    if model_prior is not None:
        try:
            prior_ent = model_prior.entropy()
        except NotImplementedError or AttributeError:
            n_ent_samples = self.n_prior if self.n_prior > int(1e6) else int(1e6)
            ent_samples = model_prior.sample( (n_ent_samples,) )
            prior_ent = -model_prior.log_prob(ent_samples).sum(0) / n_ent_samples
            del ent_samples
    else:
        raise NotImplementedError('Monte Carlo estimate of prior entropy not yet implemented')

    model_space_samples = torch.tensor(dataframe['prior'][:N+M]).float()
    
    if interrogation_mapping is not None:
        model_space_samples = interrogation_mapping(model_space_samples)
    else:
        model_space_samples = model_space_samples
    
    eig_list = []
    if return_guide:
        guide_collection = []
    losses_collection = []
    

    for design_i in tqdm(design_list, total=len(design_list), disable=disable_tqdm):

        if set_rseed:
            pyro.set_rng_seed(0)
            torch.manual_seed(0)
                                
        # design restriction need to be applied on axis one as axis 0 is the sample dimension 
        # tolist necessary as as single elemet tensor will return an int instead of a tensor
        # None necessary to make make sure the tensor is in the right shape
        N_fwd_samples         = torch.tensor(dataframe['data'][:N, design_i.tolist()])[:, None, :].float()
        M_fwd_samples         = torch.tensor(dataframe['data'][N:N+M, design_i.tolist()])[:, None, :].float()
        # generate noisy samples for each fwd model sample
        N_likeliehood = self.data_likelihood(N_fwd_samples, self.get_designs()[design_i.tolist()])
        M_likeliehood = self.data_likelihood(M_fwd_samples, self.get_designs()[design_i.tolist()])
        
        N_likeliehood_samples = N_likeliehood.sample([1, ]).flatten(start_dim=0, end_dim=1)   
        M_likeliehood_samples = M_likeliehood.sample([1, ]).flatten(start_dim=0, end_dim=1)  
        
        N_processed_samples = DataPrepocessor(N_likeliehood_samples, model_space_samples[ :N  ])      
        M_processed_samples = DataPrepocessor(M_likeliehood_samples, model_space_samples[N:N+M])      
                
        sample_loader = DataLoader(M_processed_samples, batch_size=n_batch, shuffle=True)
        
        guide = guide_template(M_likeliehood_samples, model_space_samples[N:N+M], guide_args)
        optimizer = optimizer_constructor(guide)
        if scheduler is not None:
            scheduler = scheduler_constructor(optimizer)
        
        losses = []

        for e in range(n_epochs):
            
            batch_losses = []
            for ix, (data_samples, model_samples) in enumerate(sample_loader):
                                
                ln_p_x2_given_x1 = guide.log_prob(model_samples, data_samples)
                
                # print('ln_p_x2_goven_x1', ln_p_x2_given_x1.shape)
                
                loss = -(ln_p_x2_given_x1).mean()
                
                # print('loss', loss.shape)
                
                # set_to_none slightly faster than zero_grad
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                guide.clear_cache()

                losses.append(loss.detach().item())
                del loss
                
        marginal_lp = guide.log_prob(N_processed_samples.model, N_processed_samples.data).detach()
        guide.clear_cache()
                    
        eig = prior_ent + (marginal_lp.sum(0) / N)
        
        # print(marginal_lp.shape)
        # print(eig.shape)

        eig_list.append(eig.detach().item())                   
        
        losses_collection.append(losses)
        if return_guide:
            guide_collection.append(guide)    
        # del guide 
        
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
                
        for i, mu_bias in enumerate((((model_sample[:self.K])-self.ms_means)/self.ms_stds).flatten()):
            nn.init.constant_(self.h_mu.bias[i], mu_bias)
                
        self.h_sigma = nn.Linear(self.nn_layers_args[-1], self.K*self.model_dim)
        
        # for i, std_bias in enumerate(self.ms_stds.repeat(self.K).flatten()):
        #     nn.init.constant_(self.h_sigma.bias[i], std_bias/self.K)
        
        
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
        # sqrt for consistency with multivariate gaussian dist
        comp = dist.Normal(mu , torch.sqrt(sigma)).to_event(1)
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
        
        self.h_alpha = nn.Linear(self.nn_layers_args[-1], self.K)
        
        self.h_mu =    nn.Linear(self.nn_layers_args[-1], self.K*self.model_dim) 
                
        for i, mu_bias in enumerate((((model_sample[:self.K])-self.ms_means)/self.ms_stds).flatten()):
            nn.init.constant_(self.h_mu.bias[i], mu_bias)
                
        self.h_cov   = nn.Linear(self.nn_layers_args[-1], self.K*self.model_dim*self.model_dim)
        
    def evaluate(self, data_samples):
        
        batch_dim = data_samples.shape[0]
               
        x = data_samples.clone()
                
        x -= self.ds_means
        x /= self.ds_stds        
        
        z_h = self.nn(x)
        
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
    
    def __init__(self, data_samples, model_sample, guide_args={}):
        super(Cond_Spline_NF, self).__init__()
        
        self.design_dim = data_samples.shape[-1]
        self.model_dim  = model_sample.shape[-1]
        
        self.ds_means = data_samples.mean(dim=0)
        self.ds_stds  = data_samples.mean(dim=0)
        
        self.ms_means = model_sample.mean(dim=0)
        self.ms_stds  = model_sample.std(dim=0)
        
        # if dim is prvided tuple is returned
        self.ms_min = torch.min(model_sample, dim=0)[0]
        self.ms_max = torch.max(model_sample, dim=0)[0]
        self.ms_range = self.ms_max - self.ms_min
        # add small buffer to avoide numerical issues
        self.ms_min -= 1e-1 * self.ms_range
        self.ms_max += 1e-1 * self.ms_range
        
        self.guide_args = guide_args
        
        dist_base_m = dist.Normal(torch.zeros(self.model_dim),
                                  torch.ones(self.model_dim))
        
        self.transform_list = []
        
        if 'flow_args' not in guide_args:
            guide_args['flow_args'] = [{}] * len(guide_args['flows'])
        
        if 'flows' in guide_args: 
            for i_f, flow in enumerate(guide_args['flows']):
                if 'flow_args' in guide_args:            
                    self.transform_list.append(flow(self.model_dim, context_dim=self.design_dim, **guide_args['flow_args'][i_f]))
                else:
                    self.transform_list.append(flow(self.model_dim, context_dim=self.design_dim))

        else:
           self.transform_list.append(
               transforms.conditional_spline(self.model_dim, context_dim=self.design_dim, count_bins=8))
        
        self.dist_x2_given_x1 = dist.ConditionalTransformedDistribution(dist_base_m, self.transform_list)
        
    def log_prob(self, model_space, data_samples):
        
        d = data_samples.detach().clone()
        m = model_space.detach().clone()[:, None, :]
        
        # print(d.shape)
        # print(m.shape)
        
        #TODO squeeze here is very much a hack           
        dit = self.dist_x2_given_x1.condition(d)
        
        if 'processing' in self.guide_args:
            if self.guide_args['processing'] == 'normalise':
                    # event_dim necessary for some transformations eg spline        
                    dit  = dist.TransformedDistribution(
                    dit ,  torch_transforms.AffineTransform(self.ms_means, self.ms_stds, event_dim=1))
            #TODO not working for now
            elif self.guide_args['processing'] == 'log_transform':
                # should have the same effect as the one used in Zhang 2021: Seismic Tomography Using Variational Inference Methods
                dit = dist.TransformedDistribution(
                    dit,  torch_transforms.SigmoidTransform())
                dit = dist.TransformedDistribution(
                    dit,  torch_transforms.AffineTransform(self.ms_min, (self.ms_max - self.ms_min), event_dim=1))
        
        return dit.log_prob(m)

    def parameters(self):
        return torch.nn.ModuleList(
           self.transform_list).parameters()

    def clear_cache(self):
        self.dist_x2_given_x1.clear_cache()
    
    def sample(self, data_samples):
        #TODO make sampling a bit more transparent with how many samples should be drawn from each posterior individually
        d = data_samples.detach().clone()
        dit = self.dist_x2_given_x1.condition(d)
                
        sample = dit.sample(data_samples.shape[:-1])
        return sample

