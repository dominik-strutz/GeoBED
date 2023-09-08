import os
import pickle
import time
import contextlib
from typing import Union

import torch
import torch.distributions as dist
from torch import Tensor
from torch.distributions import Distribution

import warnings
from tqdm import TqdmExperimentalWarning
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

from tqdm.autonotebook import tqdm

import dill
dill.settings['recurse'] = True # allow pickling of functions and classes
from mpire import WorkerPool as Pool
import joblib
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
set_loky_pickler('dill')

import logging

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
    'BED',
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
            return self.nuisance_dist.expand(x.shape[:-1])

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
        if type(sample_shape) == int:
            sample_shape = torch.Size([sample_shape,])        
        shape = sample_shape + self.batch_shape
        n_samples = torch.prod(torch.tensor(shape))
        out = []
        for i in range(n_samples):
            out.append(next(self._sample_generator))
        out = torch.stack(out) 
        out = out.reshape(list(sample_shape) + list(self.batch_shape) + list(self.samples.shape[-1:]))
        return out.squeeze(0)
    
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
    
    

class BED():
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
    
    def get_m_prior_dist_entropy(self) -> Tensor:
        """
        Returns the entropy of the model parameter prior distribution.
        
        Returns:
            The entropy of the model parameter prior distribution.
        """
        return self.m_prior_dist_entropy
    
    def get_m_prior_samples(
        self,
        sample_shape: Union[int, tuple] = 1,
        random_seed: int = None,
        ) -> Tensor:
        r"""
        Returns samples from the model paramete prior distribution.
        
        Arguments:
            sample_shape: The number of samples to return. If an integer is provided, the shape is (sample_shape, dim_model_parameters). If a tuple is provided, the shape is (\*sample_shape, dim_model_parameters). Defaults to 1.
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
            sample_shape: The number of samples to return. If an integer is provided, the shape is (sample_shape, n_model_parameters, dim_nuisance_parameters). If a tuple is provided, the shape is (\*sample_shape, n_model_parameters, dim_nuisance_parameters). Defaults to 1.
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
            
        return self.nuisance_dist(model_samples).sample(sample_shape)

    def get_target_prior_samples(
        self,
        sample_shape: Union[int, tuple],
        random_seed_model: int = None,
        ) -> Tensor:
        r'''
        Returns samples from the target distribution.
        
        Args:
            sample_shape: The number of samples to return. If an integer is provided, the shape is (sample_shape, dim_target_parameters). If a tuple is provided, the shape is (\*sample_shape, dim_target_parameters).
            random_seed_model: The random seed to use for sampling from the prior distribution. Defaults to None.
        
        Returns:
            Samples from the target distribution.
        '''
        if self.target_forward_function is None:
            raise ValueError("Target forward function is not defined")
        
        model_samples = self.get_m_prior_samples(sample_shape, random_seed_model)
        nuisance_samples = self.get_nuisance_prior_samples(model_samples)
        
        return self.target_forward_function(model_samples, nuisance_samples)
     
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
        n_samples: int = None,
        n_samples_model: int = None,
        n_samples_nuisance: int = 1,
        random_seed_model: int = None,
        random_seed_nuisance: int = None,
        return_parameter_samples: int = False,
        ) -> Union[Tensor, tuple]:
        """
        Samples model and nuisance parameters from their prior distributions and evaluates the forward function at the design and the sampled parameters.

        Args:
            design: Tensor describing the experimental design at which the forward function is evaluated.
            n_samples: Number of samples to return. If nuisance parameters are present, n_samples must be None and n_samples_model must be provided. Defaults to None.
            n_samples_model: Number of model parameter samples to return. If nuisance parameters are present, n_samples_model must be provided and n_samples must be None. Defaults to None.
            n_samples_nuisance: Number of nuisance parameter samples to return. Defaults to 1.
            random_seed_model: Random seed to use for sampling from the model parameter prior distribution. Defaults to None.
            random_seed_nuisance: Random seed to use for sampling from the nuisance parameter prior distribution. Defaults to None.
            return_parameter_samples: If True, the model and nuisance parameter used to evaluate the forward function are returned. Defaults to False.
        Returns:
            Returns a tensor of data samples of shape (n_samples, n_data_samples) or (n_samples_model, dim_data_samples) if no nuisance parameters are present and (n_samples_model, n_nuisance_samples, dim_data_samples) if nuisance parameters are present. If return_parameter_samples is True, a tuple of tensors is returned. The first element is the tensor of data samples. The second element is a tensor of model parameter samples of shape (n_samples_model, dim_model_parameters). The third element is a tensor of nuisance parameter samples of shape (n_samples_model, n_nuisance_samples, dim_nuisance_parameters) if nuisance parameters are present and None otherwise.
        """
        
        
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
        ) -> Union[Distribution, tuple]:
        """
        Samples model and nuisance parameters from their prior distributions, evaluates the forward function at the design and the sampled and returns the data noise distribution.

        Args:
            design: Tensor describing the experimental design at which the forward function is evaluated.
            n_samples: Number of samples to return. If nuisance parameters are present, n_samples must be None and n_samples_model must be provided. Defaults to None.
            n_samples_model: Number of model parameter samples to return. If nuisance parameters are present, n_samples_model must be provided and n_samples must be None. Defaults to None.
            n_samples_nuisance: Number of nuisance parameter samples to return. Defaults to 1.
            random_seed_model: Random seed to use for sampling from the model parameter prior distribution. Defaults to None.
            random_seed_nuisance: Random seed to use for sampling from the nuisance parameter prior distribution. Defaults to None.
            return_parameter_samples: If True, the model and nuisance parameter used to evaluate the forward function are returned. Defaults to False.

        Returns:
            Returns a distribution of data samples. If return_parameter_samples is True, a tuple is returned. The first element is the distribution of data samples. The second element is a tensor of model parameter samples of shape (n_samples_model, dim_model_parameters). The third element is a tensor of nuisance parameter samples of shape (n_samples_model, n_nuisance_samples, dim_nuisance_parameters) if nuisance parameters are present and None otherwise.

        """
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
        ) -> Union[Tensor, tuple]:
        """
        Samples model and nuisance parameters from their prior distributions, evaluates the forward function at the design and the sampled and returns samples from the data noise distribution.
        
        Args:
            design: Tensor describing the experimental design at which the forward function is evaluated.
            n_samples: Number of samples to return. If nuisance parameters are present, n_samples must be None and n_samples_model must be provided. Defaults to None.
            n_samples_model: Number of model parameter samples to return. If nuisance parameters are present, n_samples_model must be provided and n_samples must be None. Defaults to None.
            n_samples_nuisance: Number of nuisance parameter samples to return. Defaults to 1.
            n_likelihood_samples: Number of samples to return from the data noise distribution. Defaults to 1.
            random_seed_model: Random seed to use for sampling from the model parameter prior distribution. Defaults to None.
            random_seed_nuisance: Random seed to use for sampling from the nuisance parameter prior distribution. Defaults to None.
            random_seed_likelihood: Random seed to use for sampling from the data noise distribution. Defaults to None.
            return_parameter_samples: If True, the model and nuisance parameter used to evaluate the forward function are returned. Defaults to False.
            return_distribution: If True, the data noise distribution is returned. Defaults to False.
        
        Returns:
            Returns a tensor of data samples of shape (n_likelihood_samples, \*n_nuisance_samples, \*n_model_samples, dim_data_samples) if nuisance parameters are present and (n_likelihood_samples, \*n_model_samples, dim_data_samples) if no nuisance parameters are present. Nuisance parameters dimensions are squeezed if they are of size 1. If return_parameter_samples is True, a tuple of tensors is returned. The first element is the tensor of data samples. The second element is a tensor of model parameter samples of shape (n_samples_model, dim_model_parameters). The third element is a tensor of nuisance parameter samples of shape (n_samples_model, n_nuisance_samples, dim_nuisance_parameters) if nuisance parameters are present and None otherwise. If return_distribution is True, a tuple of tensors is returned. The first element is the tensor of data samples. The second element is the data noise distribution. The third element is a tensor of model parameter samples of shape (n_samples_model, dim_model_parameters). The fourth element is a tensor of nuisance parameter samples of shape (n_samples_model, n_nuisance_samples, dim_nuisance_parameters) if nuisance parameters are present and None otherwise. If both return_parameter_samples and return_distribution are True, the model and nuisance parameter samples are returned last.
        """

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
        parallel_library: str='joblib',
        random_seed: int=None,
        progress_bar: bool=True,
        ) -> tuple:
        """
        Returns the expected information gain (EIG) of a design or a list of designs. The EIG is calculated using the method specified by eig_method. The method must be one of the following:
            - 'nmc': Nested Monte Carlo
            - 'dn': Doubly Nested Monte Carlo
            - 'laplace': Laplace approximation
            - 'variational_marginal': Variational marginal
            - 'variational_marginal_likelihood': Variational marginal likelihood
            - 'variational_posterior': Variational posterior
            - 'minebed': Mutual information neural estimator
            - 'nce': Noise contrastive estimation mutual information estimator
            - 'flo': Fenchel-Legendre Optimization mutual information estimator
        
        For more information on the methods, see the documentation of the methods in the :mod:`geobed.eig_methods` module.
        
        Args:
            design: A tensor of shape (dim_design) describing a single design or a tensor of shape (n_designs, dim_design) describing a list of designs.
            eig_method: The method used to calculate the EIG. Must be one of the methods listed above.
            eig_method_kwargs: A dictionary of keyword arguments passed to the EIG method.
            filename: The filename of a file to save the results to. If the file exists, the results are loaded from the file. If the file does not exist, the results are calculated and saved to the file. Defaults to None.
            num_workers: The number of workers to use for parallelisation. Defaults to 1.
            parallel_library: The parallel library to use. Must be either 'mpire' or 'joblib'. Defaults to 'joblib'.
            random_seed: The random seed to use for sampling from the model and nuisance parameter prior distributions as well as the data noise distribution. Defaults to None.
            progress_bar: If True, a progress bar is shown. Defaults to True.
        """
        
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
            
        if eig_method == 'nmc':
            from .eig_methods.nmc import nmc as eig_calculator
        elif eig_method == 'dn':
            from .eig_methods.dn import dn as eig_calculator                                        
        elif eig_method == 'laplace':
            from .eig_methods.laplace import laplace as eig_calculator
        elif eig_method == 'variational_marginal':
            from .eig_methods.variational_marginal import variational_marginal as eig_calculator
        elif eig_method == 'variational_marginal_likelihood':
            from .eig_methods.variational_marginal_likelihood import variational_marginal_likelihood as eig_calculator
        elif eig_method == 'variational_posterior':
            from .eig_methods.mi_lower_bounds import variational_posterior as eig_calculator
        elif eig_method == 'minebed':
            from .eig_methods.mi_lower_bounds import minebed as eig_calculator
        elif eig_method == 'nce':
            from .eig_methods.mi_lower_bounds import nce as eig_calculator
        elif eig_method == 'flo':
            from .eig_methods.mi_lower_bounds import flo as eig_calculator                    
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

