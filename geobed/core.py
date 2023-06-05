from types import MethodType
import os
import pickle
import time

import numpy as np
import torch
from torch import Tensor, Size
from torch.distributions import Distribution

import dill
dill.settings['recurse'] = True # allow pickling of functions and classes

from mpire import WorkerPool as Pool
import joblib
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler

set_loky_pickler('dill')

from tqdm.autonotebook import tqdm

if 'threads_set' not in locals():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    threads_set = True

class BED_discrete(object):
    
    # import all methods from submodules her get assign them to the class
    from .design2data_helpers import lookup_1to1_fast, lookup_interstation_design, lookup_1to1_design_flexible, constructor_1to1_fast
    from .eig import nmc, dn, variational_marginal, variational_posterior, minebed, nce, FLO
    from .optim import iterative_construction
    
    def __init__(self, design_dicts, data_likelihood, prior_samples, prior_dist=None, design2data='lookup_1to1_design_flexible', verbose=True):
        
        self.design_dicts = design_dicts
        self.data_likelihood = data_likelihood
        self.prior_dist = prior_dist
        
        if not torch.is_tensor(prior_samples):
           torch.tensor(prior_samples) 
        self.prior_samples = prior_samples
        
        if prior_dist is None:
            if verbose: print('No prior distribution given. Variational posterior will only be accurate up to a constant.\n \
                  This is fine for the optimisation but just a heads up.')
        
        #TODO: keep updated
        self.design2data_methods = ['lookup_1to1_fast', 'lookup_interstation_design', 'lookup_1to1_design_flexible', 'constructor_1to1_fast']        
        self.eig_methods = ['nmc', 'dn', 'variational_marginal', 'variational_posterior', 'minebed', 'nce', 'FLO']
        self.parallel_methods = ['mpire', 'joblib']
        self.optim_methods = ['iterative_construction']
        
        if isinstance(design2data, str):
            if design2data in self.design2data_methods:
                self.design2data = getattr(self, design2data)
            else:
                raise ValueError(f'Unknown design2data method: {design2data}')
        else:            
            self.design2data = MethodType(design2data, self)
            # print('Set parallel method to mpire to use dill.')
                        
        self.n_prior = prior_samples.shape[0]
        self.name_list = list(design_dicts.keys())
            
        if self.prior_dist == None:
            if verbose: print('No prior distribution defined. Setting prior entropy to 0. This has no effect on the design optimisation.')
            self.prior_entropy = torch.tensor(0.)
        else:
            try:
                self.prior_entropy = self.prior_dist.entropy()
            except:
                if verbose: print('Entropy of prior distribution could not be calculated. Calculating it numerically.\
                       Any errors will have no effect on the design optimisation.')
                n_ent_samples = self.n_prior if self.n_prior > int(1e6) else int(1e6)
                ent_samples = self.prior_dist.sample( (n_ent_samples,) )
                self.prior_entropy = -self.prior_dist.log_prob(ent_samples).sum(0) / n_ent_samples
                del ent_samples
                
        self.verbose = verbose
            
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
            raise ValueError(f'Only {self.n_prior} samples are available, but {n_samples} were requested.')
                
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
    
    
    def calculate_eig(self, design, method, method_kwargs={}, random_seed=923981, filename=None):
        
        #for some reason manual seed zero results in extremely weird behaviour
        
        #TODO: make eig_method constistent in naming
        #TODO: write check if prior and data likelihood have the right event shape
        
        if filename is not None:
            if os.path.isfile(filename):
                if self.verbose: print(f'Loading eig values from {filename}')
                with open(filename, 'rb') as f:
                    out = pickle.load(f)
                return out
            else:
                if self.verbose: print(f'File {filename} does not exist. Calculating eig value.')
        
        start_time = time.perf_counter()
                
        if random_seed is not None:
            torch.manual_seed(random_seed)
        
        if method in self.eig_methods:
            eig_calculator = getattr(self, method)
        else:
            raise ValueError(f'Unknown eig method: {method}. Choose from {self.eig_methods}')
        
        out = eig_calculator(design, **method_kwargs)
        
        end_time = time.perf_counter()
        
        # deal with nan data
        try:
            out[1]['wall_time'] = end_time - start_time
        except TypeError:
            pass
        
        if filename is not None:
            with open(filename, 'wb') as f:
                pickle.dump(out, f)
        
        return out
        
        
    def calculate_eig_list(self, design_list, method, method_kwargs={}, 
                           num_workers=1, progress_bar=False,
                           random_seed=923981, parallel_method='joblib',
                           filename=None):
        
        if filename is not None:
            if os.path.isfile(filename):
                self.verbose: print(f'Loading eig values from {filename}')
                with open(filename, 'rb') as f:
                    out = pickle.load(f)
                return out
            else:
                self.verbose: print(f'File {filename} does not exist. Calculating eig values.')
        
        if type(design_list) is np.ndarray:
            design_list = design_list.tolist()
        
        if not isinstance(design_list[0], list):
            design_list = [[d,] for d in design_list]        
        if not isinstance(method, list):
            method = [method] * len(design_list)
        if not isinstance(method_kwargs, list):
            method_kwargs = [method_kwargs] * len(design_list)

        
        if num_workers > 1:
            
            if parallel_method == 'mpire':
                def worker(worker_id, shared_self, design, method, method_kwargs):
                    return shared_self.calculate_eig(design, method, method_kwargs, random_seed)
                            
                # spawn method is necessary for multiprocessing to be compatibal with pytorch and windows
                with Pool(n_jobs=num_workers, start_method='spawn', shared_objects=self, use_dill=True, pass_worker_id=True) as pool:
                    results = pool.map(worker, list(zip(design_list, method, method_kwargs)),
                                    progress_bar= progress_bar, progress_bar_options={'position': 1, 'desc': 'Calculating eig',})
            
            elif parallel_method == 'joblib':
                
                import contextlib
                
                @contextlib.contextmanager
                def tqdm_joblib(tqdm_object):
                    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
                    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
                        def __call__(self, *args, **kwargs):
                            tqdm_object.update(n=self.batch_size)
                            return super().__call__(*args, **kwargs)

                    old_batch_callback = joblib.parallel.BatchCompletionCallBack
                    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
                    try:
                        yield tqdm_object
                    finally:
                        joblib.parallel.BatchCompletionCallBack = old_batch_callback
                        tqdm_object.close()
                
                def worker(design, method, method_kwargs):
                    return self.calculate_eig(design, method, method_kwargs, random_seed)

                with tqdm_joblib(tqdm(desc="Calculating eig", position=1, total=len(design_list), disable=not progress_bar)) as progress_bar:
                    results = Parallel(n_jobs=num_workers)(delayed(worker)(design, method, method_kwargs) for design, method, method_kwargs in list(zip(design_list, method, method_kwargs)))
            else:
                raise ValueError(f'Unknown parallel method: {parallel_method}. Choose from {self.parallel_methods}')
            
        else:
            results = []
            for design, method, method_kwargs in tqdm(list(zip(design_list, method, method_kwargs)), disable= not progress_bar):
                results.append(self.calculate_eig(design, method, method_kwargs))

        out = list(zip(*results))          
        out[0] = torch.stack(out[0])
        
        if filename is not None:
            with open(filename, 'wb') as f:
                pickle.dump(out, f)
            
        return out
    
    def find_optimal_design(
        self, design_point_names, design_budget,
        opt_method, eig_method,
        opt_method_kwargs={},
        eig_method_kwargs={},
        num_workers=1,
        filename=None):
        
        if filename is not None:
            if os.path.isfile(filename):
                self.verbose: print(f'Loading optimal design from {filename}')
                with open(filename, 'rb') as f:
                    out = pickle.load(f)
                return out
            else:
                self.verbose: print(f'File {filename} does not exist. Calculating optimal design.')
        
        if opt_method in self.optim_methods:
            opt_method = getattr(self, opt_method)
        else:
            raise ValueError(f'Unknown opt method: {opt_method}. Choose from {self.optim_methods}')  

        out = opt_method(
            design_point_names, design_budget,
            eig_method, eig_method_kwargs,
            num_workers=num_workers,
            **opt_method_kwargs)
        
        if filename is not None:
            with open(filename, 'wb') as f:
                pickle.dump(out, f)
        
        return out