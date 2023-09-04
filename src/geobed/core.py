import os
import pickle
import time
import contextlib
from typing import Union

import torch
import torch.distributions as dist
from torch import Tensor
from torch.distributions import Distribution

from tqdm.autonotebook import tqdm

import dill
dill.settings['recurse'] = True # allow pickling of functions and classes
from mpire import WorkerPool as Pool
import joblib
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
set_loky_pickler('dill')

import logging

from . import eig_methods as EIG_METHODS

if 'THREADS_SET' not in locals():
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    THREADS_SET = True

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

__all__ = [
    'BED_Class',
]

class Delta(Distribution):
    """ Inspired by https://pytorch.org/rl/_modules/torchrl/modules/distributions/continuous.html#Delta """
    def __init__(self, loc):
        super().__init__(validate_args=False)
        self.loc = loc
    def sample(self, size=torch.Size()):
        if size is None:
            size = torch.Size([])
        return self.loc.expand(*size, *self.loc.shape)

class Dummy_Foward_Class():
    def __init__(self, forward_function):
        self.forward_function = forward_function
    def forward(self, design, model_samples, nuisance_samples=None):
        if nuisance_samples is None:
            return {'data': self.forward_function(design, model_samples)}
        else:
            return {'data': self.forward_function(design, model_samples, nuisance_samples)}

class dummy_cond_nuisance_dist():
    def __init__(self, nuisance_dist):
        self.nuisance_dist = nuisance_dist

    def __call__(self, x):
        if x is None:
            return self.nuisance_dist
        else:
            return self.nuisance_dist.expand((x.shape[0],))

class Obs_Noise_Dist_Wrapper():
    def __init__(self, obs_noise_dist):
        self.obs_noise_dist = obs_noise_dist
    def __call__(self, fwd_out_dict, design):        
        data_samples = fwd_out_dict['data']
        return self.obs_noise_dist(data_samples, design=design, **fwd_out_dict)


class SampleDistribution():
    def __init__(self, samples: Tensor):
        self.samples = samples
        self.iterator_counter = 0
        self._sample_generator = self._sample_generator()
        
        self.batch_shape = torch.Size([])
        
    def sample(self, sample_shape: torch.Size([])):
        
        shape = sample_shape + self.batch_shape
                
        n_samples = torch.prod(torch.tensor(shape))
        out = []
        for i in range(n_samples):
            out.append(next(self._sample_generator))
        out = torch.stack(out)
        
        return out.reshape(shape + self.samples.shape[-1:])
    
    def expand(self, batch_shape):
        self.batch_shape = batch_shape
        return self
    
    def _sample_generator(self):
        while True:
            yield self.samples[self.iterator_counter]
            self.iterator_counter += 1
            if self.iterator_counter == self.samples.shape[0]:
                raise ValueError('Not enough prior samples avaliable.')
            
    def entropy(self):
        logging.info('Entropy of sample distribution is not defined. Setting it to zero.')
        return torch.tensor(0.)
        
    def log_prob(self, x):
        raise NotImplementedError('Log probability of sample distribution is not defined.')
    
    

class BED_Class():
    r"""
    A class that represents a Bayesian experimental design problem. It contains the forward function, the prior distributions of the model and nuisance parameters, and the observation noise distribution. It also contains methods to calculate the expected information gain (EIG) of a design.
        
    Arguments:
        forward_function: A function that takes in a design and model (and nuisance) parameters and returns a tensor of data samples. The output can either be a tensor of data samples or a dictionary with the key 'data' and value being a tensor of data samples. The shape of the tensor of data samples should be either (n_model_samples, n_nuisance_samples, n_data_samples) or (n_model_samples, n_data_samples) if no nuisance parameters are present.
        m_prior_dist: A distribution of model parameters. If a tensor is provided, it is assumed to be a set of samples from the distribution. The distribution is only required to have a sample method. A log_prob method is required for some EIG methods and for the calculation of the entropy of the prior distribution if it is not provided.
        nuisance_dist: A distribution of nuisance parameters. Can be a dependent distribution. If a tensor is provided, it is assumed to be a set of samples from the distribution. The distribution is only required to have a sample method. A log_prob method is required for some EIG methods and for the calculation of the entropy of the nuisance distribution if it is not provided. If a callable is provided, it is assumed to be a function that takes in a tensor of model parameters and returns a conditional distribution of nuisance parameters.
        obs_noise_dist: A distribution of observation noise. Takes in a tensor of data samples and returns a distribution of data samples. If None, a delta distribution is used, which is equivalent to no observation noise and indicates an implicit observation noise distribution through the forward function. Is required for some EIG methods.
        target_forward_function: A function that takes in a design and model parameters and returns a tensor of data samples. Is used for interrogation problems in which a function of the model (and nuisance) parameters is of interest. If None, the function corresponds to a identity function of the model parameters.
    """
    # The __init__ method may be documented in either the class level
    # docstring, or as a docstring on the __init__ method itself.
    def __init__(
        self,
        forward_function: callable,
        m_prior_dist: Distribution,
        nuisance_dist: Union[Distribution, callable] = None,
        obs_noise_dist: callable = None,
        target_forward_function: callable = None,
        ):
        if callable(forward_function):   
            self.forward_function = Dummy_Foward_Class(forward_function)
        else:
            self.forward_function = forward_function
            
        #  Check if either prior samples or prior distribution is provided
        if type(m_prior_dist) == torch.Tensor:
            self.m_prior_dist = SampleDistribution(m_prior_dist)
        else:
            self.m_prior_dist = m_prior_dist
        
        # Enforce that nuisance distribution might be conditional on model parameters
        if not callable(nuisance_dist) and nuisance_dist is not None:
            # Check if nuisance samples or nuisance distribution is provided
            if type(nuisance_dist) == torch.Tensor:
                nuisance_dist = SampleDistribution(nuisance_dist)
            
            self.nuisance_dist = dummy_cond_nuisance_dist(nuisance_dist)
        else:
            self.nuisance_dist = nuisance_dist        
        
        try:
            self.m_prior_dist_entropy = self.m_prior_dist.entropy()
        except:
            try:
                ent_samples = self.m_prior_dist.sample( (int(1e6),) )
                self.m_prior_dist_entropy = -self.m_prior_dist.log_prob(ent_samples).sum(0) / int(1e6)
                logging.info('''Entropy of prior distribution could not be calculated. Calculating it numerically. 
                            Any errors will have no effect on the design optimisation.''')
                del ent_samples
            except:
                logging.info('''Entropy of prior distribution could not be calculated. Setting it to zero. 
                            This will have no effect on the design optimisation.''')
                self.m_prior_dist_entropy = torch.tensor(0.)
        
        # Check if observation noise distribution is provided
        # Check if observation noise distribution is explicitly or implicitly defined
        if obs_noise_dist is None:
            obs_noise_dist = lambda x: Delta(x)
            self.implict_obs_noise_dist = True
        else:
            self.implict_obs_noise_dist = False
            
        self.obs_noise_dist = Obs_Noise_Dist_Wrapper(obs_noise_dist)
        
        self.target_forward_function = target_forward_function
                    
        self.eig_methods = [
            'nmc', 'dn',
            'laplace',
            'variational_marginal',
            'variational_marginal_likelihood',
            'variational_posterior', 'minebed', 'nce', 'flo']
    
    def get_m_prior_samples(
        self,
        sample_shape: Union[int, tuple] = 1,
        random_seed: int = None,
        ) -> Tensor:
        r"""
        Returns samples from the model paramete prior distribution.
        
        Arguments:
            sample_shape: The number of samples to return. If an integer is provided, the shape is (sample_shape, n_model_parameters). If a tuple is provided, the shape is (\*sample_shape, n_model_parameters). Defaults to 1.
            random_seed: The random seed to use for sampling from the prior distribution. Defaults to None.
        
        Returns:
            Samples from the model parameter prior distribution.
        """    
        if type(sample_shape) == int:
            sample_shape = (sample_shape,)
        
        if random_seed is not None:
            torch.manual_seed(random_seed)

        return self.m_prior_dist.sample(sample_shape)
        
    def get_nuisance_prior_samples(
        self,
        model_samples: Tensor,
        sample_shape: Union[int, tuple] = 1,
        random_seed: int = None,
        ) -> Tensor:

        r"""
        This function returns samples from the nuisance parameter prior distribution (conditional on the model parameters).
        
        Args:
            model_samples: Samples from the model parameter prior distribution on which the nuisance parameter prior distribution is conditioned if it is dependent. 
            sample_shape: The number of samples to return. If an integer is provided, the shape is (sample_shape, n_nuisance_parameters). If a tuple is provided, the shape is (\*sample_shape, n_nuisance_parameters). Defaults to 1.
            random_seed: The random seed to use for sampling from the prior distribution. Defaults to None.
        
        Returns:
            Samples from the nuisance parameter prior distribution (conditional on the model parameters).
        """

        if self.nuisance_dist is None:
            return None
        
        if type(sample_shape) == int:
            sample_shape = (sample_shape,)
        
        if random_seed is not None:
            torch.manual_seed(random_seed)
            
        return self.nuisance_dist(model_samples).sample(sample_shape).swapaxes(0,1)

    def get_target_prior_samples(
        self,
        sample_shape: Union[int, tuple],
        random_seed_model: int = None,
        ) -> Tensor:
        r'''
        Returns samples from the target distribution.
        
        Args:
            sample_shape: The number of samples to return. If an integer is provided, the shape is (sample_shape, n_target_parameters). If a tuple is provided, the shape is (\*sample_shape, n_target_parameters).
            random_seed_model: The random seed to use for sampling from the prior distribution. Defaults to None.
        
        Returns:
            Samples from the target distribution.
        '''
        if self.target_forward_function is None:
            raise ValueError("Target forward function is not defined")
        
        return self.target_forward_function(self.get_m_prior_samples(sample_shape, random_seed_model))
     
    def _get_forward_function_samples(
        self,
        design: Tensor,
        n_samples: int=None,
        n_samples_model: int=None,
        n_samples_nuisance: int=1,
        random_seed_model=None,
        random_seed_nuisance=None,
        ) -> Tensor:

        if self.nuisance_dist is None:
            if (n_samples_model is None) and (n_samples is None):
                raise ValueError("Either n_samples_model or n_samples must be provided")
            elif (n_samples_model is not None) and (n_samples is not None):
                raise ValueError("Either n_samples_model or n_samples must be provided")
            elif n_samples_model is None:
                n_samples_model = n_samples
         
        else:
            if (n_samples is not None):
                raise ValueError("n_samples cannot be provided if nuisance parameters are present")
            if (n_samples_model is None):
                raise ValueError("n_samples_model must be provided")
            
        model_samples = self.get_m_prior_samples(n_samples_model, random_seed_model)
        nuisance_samples = self.get_nuisance_prior_samples(model_samples, n_samples_nuisance, random_seed_nuisance)
                
        if nuisance_samples is None:
            fwd_out = self.forward_function.forward(design, model_samples)
        else:
            fwd_out = self.forward_function.forward(design, model_samples, nuisance_samples=nuisance_samples)     
    
        out = {}
        # extract data samples
        if type(fwd_out) == dict:
            out.update(fwd_out)
        else:
            out['data'] = fwd_out

        # flatten nuisance dimension if it is of size 1
        if out['data'].ndim == 3:
            out['data'] = out['data'].squeeze(1)
        
        out['model_samples'] = model_samples
        out['nuisance_samples'] = nuisance_samples        
        return out
    
    def get_forward_function_samples(
        self,
        design: Tensor,
        n_samples: int=None,
        n_samples_model: int=None,
        n_samples_nuisance: int=1,
        random_seed_model=None,
        random_seed_nuisance=None,
        return_parameter_samples=False,
        ) -> Tensor:
        
        out = self._get_forward_function_samples(
                design, n_samples, n_samples_model, n_samples_nuisance, random_seed_model, random_seed_nuisance)
        
        if return_parameter_samples:
            return out['data'], out['model_samples'], out['nuisance_samples']
        else:
            return out['data']

        
    def get_foward_model_distribution(
        self,        
        design: Tensor,
        n_samples: int=None,
        n_samples_model: int=None,
        n_samples_nuisance: int=1,
        random_seed_model=None,
        random_seed_nuisance=None,
        return_parameter_samples=False,
        ) -> Distribution:
        
        fwd_out = self._get_forward_function_samples(
            design,
            n_samples,
            n_samples_model,
            n_samples_nuisance,
            random_seed_model, random_seed_nuisance,
            )

        if return_parameter_samples:
            return self.obs_noise_dist(
                fwd_out,
                design=design), fwd_out['model_samples'], fwd_out['nuisance_samples']
                
        else:
            return self.obs_noise_dist(
                fwd_out,
                design=design)

    def get_forward_model_samples(
        self,
        design: Tensor,
        n_samples: int=None,
        n_samples_model: int=None,
        n_samples_nuisance: int=1,
        n_likelihood_samples: int=1,
        random_seed_model=None,
        random_seed_nuisance=None,
        random_seed_likelihood=None,
        return_parameter_samples=False,
        return_distribution=False,
        ) -> Tensor:
    
        if type(n_likelihood_samples) == int:
            n_likelihood_samples = (n_likelihood_samples,)    
        
        out = []
        
        if not return_parameter_samples:
            dist = self.get_foward_model_distribution(
                design, n_samples, n_samples_model, n_samples_nuisance, random_seed_model, random_seed_nuisance,)
        else:
            dist, model_samples, nuisance_samples = self.get_foward_model_distribution(
                design, n_samples, n_samples_model, n_samples_nuisance, random_seed_model, random_seed_nuisance, return_parameter_samples=True)
            
        if random_seed_likelihood is not None:
            torch.manual_seed(random_seed_likelihood)
            
        if not return_distribution and not return_parameter_samples:
            return dist.sample(n_likelihood_samples).squeeze(0)
        elif not return_distribution and return_parameter_samples:
            return dist.sample(n_likelihood_samples).squeeze(0), model_samples, nuisance_samples
        elif return_distribution and not return_parameter_samples:
            return dist.sample(n_likelihood_samples).squeeze(0), dist
        elif return_distribution and return_parameter_samples:
            return dist.sample(n_likelihood_samples).squeeze(0), dist, model_samples, nuisance_samples
        
    
    def calculate_EIG(
        self,
        design: Tensor,
        eig_method: str,
        eig_method_kwargs: dict,
        filename: str=None,
        num_workers: int=1,
        parallel_library: str='mpire',
        random_seed=None,
        progress_bar: bool=True,
        ) -> Tensor:
        
        if design.ndim == 1:
            design = design.unsqueeze(0)
        if not isinstance(eig_method, list):
            eig_method = [eig_method] * len(design)
        if not isinstance(eig_method_kwargs, list):
            eig_method_kwargs = [eig_method_kwargs] * len(design)
                        
        if filename is not None:
            if os.path.isfile(filename):
                logging.info(f'Loading results from {filename}')
                with open(filename, 'rb') as f:
                    out = pickle.load(f)
                return out
            else:
                logging.info(f'File {filename} does not exist. Calculating results.')
        
        results = []
        if num_workers == 1 or len(design) == 1:
            if len(design) == 1:
                progress_bar = False
            
            for i, d in tqdm(enumerate(design), disable=not progress_bar, desc='Calculating eig', position=1, total=len(design)):
                out = self._calculate_EIG_single(d, eig_method[i], eig_method_kwargs[i], random_seed)
                results.append(out)
                del out
        else:
            if parallel_library == 'mpire':
                def worker(worker_id, shared_self, d, m, m_kwargs):
                    return shared_self._calculate_EIG_single(d, m, m_kwargs, random_seed)
                # spawn method is necessary for multiprocessing to be compatibal with pytorch and windows
                with Pool(n_jobs=num_workers, start_method='spawn', shared_objects=self, use_dill=True, pass_worker_id=True) as pool:
                    results = pool.map(worker, list(zip(design, eig_method, eig_method_kwargs)),
                                    progress_bar= progress_bar, progress_bar_options={'position': 1, 'desc': 'Calculating eig',})
                    
            elif parallel_library == 'joblib':
                def worker(d, m, m_kwargs):
                    return self._calculate_EIG_single(d, m, m_kwargs, random_seed)

                with tqdm_joblib(tqdm(desc="Calculating eig", position=1, total=len(design), disable=not progress_bar)) as progress_bar:
                    results = Parallel(n_jobs=num_workers)(
                        delayed(worker)(design, method, method_kwargs) 
                        for design, method, method_kwargs 
                        in list(zip(design, eig_method, eig_method_kwargs)))
            
            else:
                raise ValueError(f'Unknown parallel library: {parallel_library}. Choose from mpire or joblib')
        
        results = list(zip(*results))          
        results[0] = torch.stack(results[0])
        
        if filename is not None:
            with open(filename, 'wb') as f:
                pickle.dump(results, f)
        
        return results
    
    def _calculate_EIG_single(self, design, eig_method, eig_method_kwargs, random_seed=None):
        
        eig_method = eig_method.lower()
        if eig_method in self.eig_methods:
                                    
            eig_calculator = getattr(getattr(EIG_METHODS, eig_method), eig_method)
            
        else:
            raise ValueError(f'Unknown eig method: {eig_method}. Choose from {self.eig_methods}')                                

        start_time = time.perf_counter()

        if random_seed is not None: torch.manual_seed(random_seed)
        out = eig_calculator(self, design, **eig_method_kwargs)
        
        # deal with nan data
        try:
            out[1]['wall_time'] = time.perf_counter() - start_time
        except TypeError:
            pass
    
        return  out

