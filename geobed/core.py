from types import MethodType
import abc

import numpy as np
import torch
from torch import Tensor, Size
from torch.distributions import Distribution

import dill
dill.settings['recurse'] = True # allow pickling of functions and classes

from mpire import WorkerPool as Pool
from tqdm.autonotebook import tqdm

if 'threads_set' not in locals():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    threads_set = True

class BED_discrete(object):
    
    # import all methods from submodules her get assign them to the class
    from .design2data_helpers import lookup_1to1_design, lookup_interstation_design, lookup_1to1_design_variable_length
    from .eig import dn, variational_marginal, variational_posterior
    from .optim import iterative_construction
    
    def __init__(self, design_dicts, data_likelihood, prior_samples, prior_dist=None, design2data='lookup_1to1_design_variable_length'):
        
        self.design_dicts = design_dicts
        self.data_likelihood = data_likelihood
        self.prior_dist = prior_dist
        self.prior_samples = prior_samples
        
        if prior_dist is None:
            print('No prior distribution given. Variational posterior will only be accurate up to a constant.\
                  This is fine for the optimisation but just a heads up.')
        
        #TODO: keep updated
        self.design2data_methods = ['lookup_1to1_design', 'lookup_interstation_design', 'lookup_1to1_design_variable_length']        
        self.eig_methods = ['dn', 'variational_marginal', 'variational_posterior']
        self.optim_methods = ['iterative_construction']
        
        if isinstance(design2data, str):
            if design2data in self.design2data_methods:
                self.design2data = getattr(self, design2data)
            else:
                raise ValueError(f'Unknown design2data method: {design2data}')
        else:
            self.design2data = MethodType(design2data, self)
                        
        self.n_prior = prior_samples.shape[0]
        self.name_list = list(design_dicts.keys())
            
    def get_forward_samples(self, design: list, n_samples: int = None) -> Tensor:
        """Returns forward modeled samples from the prior distribution for a design point.

        Args:
            design (list): List of design point names.
            n_samples (int, optional): Number of samples to be returned. Has to be less than the total number of prior samples. Defaults to n_prior.
        Raises:
            ValueError: Raises an error if the number of samples requested is larger than the total number of prior samples.

        Returns:
            Tensor: Array of shape (n_samples, n_designs) containing the samples from the forward modeled prior distribution.
        """        
    
        n_samples = self.n_prior if n_samples is None else n_samples
        
        if n_samples > self.n_prior:
            raise ValueError(f'Only {self.n_prior} samples are available!')
                
        data = self.design2data(design, n_samples)
        
        return data
    
    
    def get_likelihoods(self, design: list, n_samples: int = None) -> Distribution:
        """Returns the data likelihood distribution for a design point.

        Args:
            design (list): List of design point names.
            n_samples (int, optional): Number of samples to be returned. Has to be less than the total number of prior samples. Defaults to n_prior.

        Returns:
            Distribution: Data likelihood distribution.
        """
        forward_samples = self.get_forward_samples(design, n_samples)
        
        if forward_samples is None:
            return None
        
        d_dicts = {n: self.design_dicts[n] for n in design}
        likelihood_kwargs = {'design': design, 'd_dicts': d_dicts}
        
        return self.data_likelihood(forward_samples, **likelihood_kwargs)
        
             
    def get_likelihood_samples(self, design: list, n_samples: int = None) -> Tensor:
        """Returns noisy forward modeled samples from the prior distribution for each design point. Noise is generated using the data likelihood function provided.

        Args:
            design (list): List of design point names.
            n_samples (int, optional): Number of samples to be returned. Has to be less than the total number of prior samples. Defaults to None.

        Returns:
            ndarray: Array of shape (n_samples, n_designs) containing the noisy samples from the forward modeled prior distribution.
        """        
        forward_samples = self.get_forward_samples(design, n_samples)
                
        if forward_samples is None:
            return None
        
        d_dicts = {n: self.design_dicts[n] for n in design}

        if self.data_likelihood is None:
            # deal with implict models where the data likelihood is not available and the forward samples are the likelihood samples
            return forward_samples
        
        likelihood_kwargs = {'design': design, 'd_dicts': d_dicts}
        likelihoods = self.data_likelihood(forward_samples, **likelihood_kwargs)
    
        samples = likelihoods.sample()
                
        return samples
    
    
    def calculate_eig(self, design, method, method_kwargs={}):
        
        #TODO: write check if prior and data likelihood have the right event shape
        
        if method in self.eig_methods:
            self.eig_calculator = getattr(self, method)
        else:
            raise ValueError(f'Unknown eig method: {method}. Choose from {self.eig_methods}')
        
        out = self.eig_calculator(design, **method_kwargs)
        
        return out
        
        
    def calculate_eig_list(self, design_list, method, method_kwargs={}, num_workers=1, progress_bar=True):
        
        # TODO: make this work fast with a single worker
        
        if not isinstance(method, list):
            method = [method] * len(design_list)
        if not isinstance(method_kwargs, list):
            method_kwargs = [method_kwargs] * len(design_list)
        
        if num_workers > 1:
            
            def worker(worker_id, design, method, method_kwargs):
                return self.calculate_eig(design, method, method_kwargs)
            
            # spawn method is necessary for multiprocessing to be compatibal with pytorch and windows
            with Pool(n_jobs=num_workers, start_method='spawn', use_dill=True, pass_worker_id=True) as pool:
                results = pool.map(worker, list(zip(design_list, method, method_kwargs)),
                                   progress_bar= progress_bar, progress_bar_options={'position': 1, 'desc': 'Calculating eig',})
                
        else:
            results = []
            for design, method, method_kwargs in tqdm(list(zip(design_list, method, method_kwargs)), disable= not progress_bar):
                results.append(self.calculate_eig(design, method, method_kwargs))

        out = list(zip(*results))
        
        return out
    
    def find_optimal_design(self, design_point_names, design_budget, opt_method, eig_method, opt_method_kwargs, eig_method_kwargs={},num_workers=1, progress_bar=True):
        
        if opt_method in self.optim_methods:
            opt_method = getattr(self, opt_method)
        else:
            raise ValueError(f'Unknown opt method: {opt_method}. Choose from {self.optim_methods}')  

        opt_method_kwargs['num_workers'] = num_workers
        opt_method_kwargs['progress_bar'] = progress_bar
                
        out = opt_method(design_point_names, design_budget, eig_method, eig_method_kwargs, **opt_method_kwargs)
        
        return out