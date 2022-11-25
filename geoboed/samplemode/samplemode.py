from fileinput import filename
import math

import numpy as np
import torch
from tqdm import tqdm

from .dn import dn
from .nmc import nmc
from .var_marg import var_marg
from .var_post import var_post

from .iterative_construction import _find_optimal_design_iterative_construction

from .dataloader import Dataloader


class BOED_sample():
    def __init__(self, filename, data_likelihood, design_independent_likeliehood=False, design_restriction=None, **kwargs):
        """_summary_

        Args:
            filename (str): Location of a file containing synthetic measurements calculated by forward modelling samples from the model prior. The file should contain a array with the name 'data', an array with the name 'designs' and an array with the name 'prior', for the measurements, the designs and the prior samples respectively. The prior samples should be in the same order as the measurements. The accepted filetypes for now are .npz and .hdf5.
            design_restriction (func, optional): Function that restrictes the designs used in the optimization in some sense. Can be used for example to downsample the design points for faster calculations with larger spacings. Defaults to None.
            model_prior (pyro.distribution or torch.distribution, optional): Distribution describing the model prior to ease the calculation of the prior entropy for the variational posterior method.. Defaults to None.

        Raises:
            ValueError: _description_
        """
        #TODO: Implement generator mode
        
        self.filename = filename
        
        if filename.endswith('.hdf5'):
            self.filetype = 'hdf5'
        elif filename.endswith('.npz'):
            self.filetype = 'npz'
        else:
            raise ValueError('Filetype not supported')
        
        if not callable(design_restriction):
            def design_restriction(x):
                return x
        self.design_restriction = design_restriction
        
        with Dataloader(filename) as dataframe:
            self.n_prior      = dataframe['prior'].shape[0]
            self.model_dim    = dataframe['prior'].shape[1]
            
            self.designs    = dataframe['design'][:]
            self.n_designs  = self.designs.shape[0]

            self.prior      = dataframe['prior'][:]

            assert self.n_prior   == dataframe['data'].shape[0], 'The number of prior samples and size of the second dimension of the data array must be the same.'
            assert self.n_designs == dataframe['data'].shape[1], 'The number of design samples and size of the first dimension of the data array must be the same.'

        self.data_likelihood = data_likelihood
        self.design_independent_likelihood = design_independent_likeliehood

    def get_designs(self):
        """Returns the designs used in the optimization.

        Returns:
            ndarray: Array of shape (n_designs, 1) containing the designs used in the optimization.
        """        
        return self.design_restriction(self.designs)
        
    def get_prior(self, n_samples=None):
        """Returns samples from the prior distribution.

        Args:
            n_samples (int, optional): Number of samples to be returned. Has to be less than the total number of prior samples. Defaults to None.

        Raises:
            ValueError: Raises an error if the number of samples requested is larger than the total number of prior samples.

        Returns:
            ndarray: Array of shape (n_samples, model_dim) containing the samples from the prior distribution.
        """        
        n_samples = self.n_prior if n_samples is None else n_samples
        if n_samples > self.n_prior:
            raise ValueError('Only {self.n_prior} samples are available!')
                     
        return self.prior[:n_samples]
    
    def get_samples(self, n_samples=None):
        """Returns forward modeled samples from the prior distribution for each design point.

        Args:
            n_samples (int, optional): Number of samples to be returned. Has to be less than the total number of prior samples. Defaults to None.

        Raises:
            ValueError: Raises an error if the number of samples requested is larger than the total number of prior samples.

        Returns:
            ndarray: Array of shape (n_samples, n_designs) containing the samples from the forward modeled prior distribution.
        """        
        
        n_samples = self.n_prior if n_samples is None else n_samples
        if n_samples > self.n_prior:
            raise ValueError(f'Only {self.n_prior} samples are available!')
        
        with Dataloader(self.filename) as dataframe:
            samples = np.apply_along_axis(self.design_restriction, 1, dataframe['data'][:n_samples])
                    
        return samples
                        
    def get_noisy_samples(self, n_samples=None):
        """Returns noisy forward modeled samples from the prior distribution for each design point. Noise is generated using the data likelihood function provided.

        Args:
            n_samples (int, optional): Number of samples to be returned. Has to be less than the total number of prior samples. Defaults to None.

        Returns:
            ndarray: Array of shape (n_samples, n_designs) containing the noisy samples from the forward modeled prior distribution.
        """        
        samples = torch.tensor(self.get_samples(n_samples))
        
        if self.design_independent_likelihood:
            samples = self.data_likelihood(samples).sample([1, ]).flatten(start_dim=0, end_dim=1)
        else:
            samples = self.data_likelihood(samples, self.designs).sample([1, ]).flatten(start_dim=0, end_dim=1)
        
        return samples.numpy()
    
    def find_optimal_design(self, design_dim,
                            optimization_method, boed_method,
                            optimization_method_kwargs={}, boed_method_kwargs={},
                            return_information=True, save_information=False,
                            n_parallel=1, **kwargs):
        """_summary_

        Args:
            boed_method (_type_): _description_
        """        

        if optimization_method == 'iterative construction':
            
            out = _find_optimal_design_iterative_construction(
                self, design_dim, boed_method, boed_method_kwargs, n_parallel=n_parallel, **optimization_method_kwargs)
            
        elif optimization_method == 'iterative decimation':
            raise NotImplementedError('Iterative decimation not implemented yet')
        elif optimization_method == 'iterative exchange':
            raise NotImplementedError('Iterative exchange not implemented yet')
        elif optimization_method == 'genetic algorithm':
            raise NotImplementedError('Genetic algorithm not implemented yet')
        elif optimization_method == 'simulated annealing':
            raise NotImplementedError('Simulated annealing not implemented yet')
        elif optimization_method == 'global search':
            raise NotImplementedError('Global search not implemented yet')
        elif optimization_method in ['variational stochastic gradient descent', 'vsgd']:
            raise NotImplementedError('Variational stochastic gradient descent not implemented yet')
        else:
            raise ValueError('Optimization method not supported')#TODO: Keep updated with new methods
        
        out = list(out)
        out[0] = self.get_designs()[out[0]]
                            
        if save_information:
            filename = f'{save_information}_oed_{optimization_method}_{boed_method}_{design_dim}'
            np.savez(filename, **out[1])
        
        if return_information:
            return out
        else:
            return out[0]   
    
    def get_eig(self, design_list, boed_method, boed_method_kwargs={}, disable_tqdm=False,
                **kwargs):
                
        if boed_method in ['dn', 'nmc', 'var_marg', 'var_post']:
            self.boed_method = boed_method
        else:
            raise ValueError('BOED method not implemented. Please choose from one of the following: dn, nmc, var_marg, var_post')
                
        with Dataloader(self.filename) as dataframe:
            if self.boed_method == 'dn':
                self.oed_results = dn(self, dataframe, design_list, disable_tqdm=disable_tqdm,
                                      **boed_method_kwargs)
            elif self.boed_method == 'nmc':
                self.oed_results = nmc(self, dataframe, design_list, disable_tqdm=disable_tqdm,
                                       **boed_method_kwargs)
            elif self.boed_method == 'var_marg':
                self.oed_results = var_marg(self, dataframe, design_list, disable_tqdm=disable_tqdm,
                                            **boed_method_kwargs)
            elif self.boed_method == 'var_post':
                self.oed_results = var_post(self, dataframe, design_list, disable_tqdm=disable_tqdm,
                                            **boed_method_kwargs)
                            
        return self.oed_results
        
        
        
