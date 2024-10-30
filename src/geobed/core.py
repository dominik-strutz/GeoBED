r""" Base classes for Bayesian experimental design methods. 


The class :class:`BED_base` defines the base class for Bayesian experimental design methods. This class can not be used directly. Instead, it is inherited by the other classes in this module. All other classes in this module inherit from this class and therefore have access to all its methods.

"""

import os
import pickle
import time
import contextlib
import inspect
from typing import Union

import torch
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

from .utils.sample_distribution import SampleDistribution
from .utils.misc import _Dummy_Cond_Dist

__all__ = [
    'BED_base', 'BED_base_explicit', 'BED_base_nuisance',
]

class BED_base():
    r"""
    Defines the base class for Bayesian experimental design methods.
    """
    
    def __init__(
        self,
        m_prior_dist: Distribution,
        ):
        r"""
        Base class for Bayesian experimental design methods. This class can not be used directly. Instead, it is inherited by the other classes in this module.
        
        Args:
            m_prior_dist: The prior distribution of the model parameters. Must be a :class:`torch.distributions.Distribution` object.
        """
        
        # Check if either prior samples or prior distribution is provided
        if type(m_prior_dist) == torch.Tensor:
            self.m_prior_dist = SampleDistribution(m_prior_dist)
        else:
            self.m_prior_dist = m_prior_dist        
        
        try:
            self.m_prior_dist_entropy = self.m_prior_dist.entropy()
        except NotImplementedError:
            pass
        
        self.m_prior_dist_entropy = None
        
        try:
            self.m_prior_dist_entropy = self.m_prior_dist.entropy()
        except NotImplementedError:
            pass
        
        if self.m_prior_dist_entropy is None:
            try:                
                self.m_prior_dist.log_prob(torch.zeros_like(self.m_prior_dist.sample()))           
                ent_samples = self.m_prior_dist.sample( (int(1e5),) )
                self.m_prior_dist_entropy = -self.m_prior_dist.log_prob(ent_samples).sum(0) / int(1e5)
                logging.info('''Entropy of prior distribution could not be calculated. Calculating it numerically. 
                            Any errors will have no effect on the design optimisation.''')
                del ent_samples
            except NotImplementedError:
                logging.info('''Entropy of prior distribution could not be calculated. Setting it to zero. 
                            This will have no effect on the design optimisation.''')
                self.m_prior_dist_entropy = torch.tensor(0.)
    
        self.eig_methods = [
            'nmc', 'dn',
            'laplace',
            'variational_marginal',
            'variational_marginal_likelihood',
            'variational_posterior', 'minebed', 'nce', 'flo']
    
    def get_m_prior_entropy(self) -> Tensor:
        """
        Returns the entropy of the model parameter prior distribution.
        
        Entropy will be calculated analytically if possible. If not, it will be calculated numerically. If neither is possible, the entropy will be set to zero, which will have no effect on the design optimisation.
        
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
        Returns samples from the model parameter prior distribution.
        
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
        Returns the expected information gain (EIG) for a design or a list of designs. The EIG is calculated using the method specified by eig_method. The method must be one of the following:
            - 'nmc': Nested Monte Carlo (:mod:`geobed.eig.nmc`)
            - 'dn': DN-Method (:mod:`geobed.eig.dn`)
            - 'laplace': Laplace approximation (:mod:`geobed.eig.laplace`)
            - 'variational_marginal': Variational marginal (:mod:`geobed.eig.variational_marginal`)
            - 'variational_marginal_likelihood': Variational marginal likelihood (:mod:`geobed.eig.variational_marginal_likelihood`)
            - 'variational_posterior': Variational posterior (:mod:`geobed.eig.variational_posterior`)
            - 'minebed': Mutual information neural estimator (:mod:`geobed.eig.minebed`)
            - 'nce': Noise contrastive estimation mutual information estimator (:mod:`geobed.eig.nce`)
            - 'flo': Fenchel-Legendre Optimization mutual information estimator (:mod:`geobed.eig.flo`)
        
        For more information on the methods, see the documentation of the methods in the :mod:`geobed.eig` module.
        
        Args:
            design: A tensor of shape (design_dim) describing a single design or a tensor of shape (n_designs, design_dim) describing a list of designs.
            eig_method: The method used to calculate the EIG. Must be one of the methods listed above.
            eig_method_kwargs: A dictionary of keyword arguments passed to the EIG method.
            filename: The filename of a file to save the results to. If the file exists, the results are loaded from the file. If the file does not exist, the results are calculated and saved to the file. Defaults to None.
            num_workers: The number of workers to use for parallelisation. Defaults to 1.
            parallel_library: The parallel library to use. Must be either 'mpire' or 'joblib'. Defaults to 'joblib'.
            random_seed: The random seed to use for sampling from the model and nuisance parameter prior distributions as well as the data noise distribution. Defaults to None.
            progress_bar: If True, a progress bar is shown. Defaults to True.
        """
                                
        if filename is not None:
            if os.path.isfile(filename):
                logging.info(f'Loading results from {filename}')
                with open(filename, 'rb') as f:
                    out = pickle.load(f)
                return out
            else:
                logging.info(f'File {filename} does not exist. Calculating results.')

        if isinstance(design, Tensor):            
            if design.ndim == 2:
                design = design.unsqueeze(0)                
        if not isinstance(eig_method, list):
            eig_method = [eig_method] * len(design)
        if not isinstance(eig_method_kwargs, list):
            eig_method_kwargs = [eig_method_kwargs] * len(design)
        
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
        if len(results[0]) == 1:
            results[0] = results[0].squeeze(0)
        
        if filename is not None:
            with open(filename, 'wb') as f:
                pickle.dump(results, f)
        
        return results
    
    def _calculate_EIG_single(self, design, eig_method, eig_method_kwargs, random_seed=None):
        
        eig_method = eig_method.lower()
            
        if eig_method == 'nmc':
            from .eig.nmc import nmc as eig_calculator
        elif eig_method == 'dn':
            from .eig.dn import dn as eig_calculator                                        
        elif eig_method == 'laplace':
            from .eig.laplace import laplace as eig_calculator
        elif eig_method == 'variational_marginal':
            from .eig.variational_marginal import variational_marginal as eig_calculator
        elif eig_method == 'variational_marginal_likelihood':
            from .eig.variational_marginal_likelihood import variational_marginal_likelihood as eig_calculator
        elif eig_method == 'variational_posterior':
            from .eig.variational_posterior import variational_posterior as eig_calculator
        elif eig_method == 'minebed':
            from .eig.minebed import minebed as eig_calculator
        elif eig_method == 'nce':
            from .eig.nce import nce as eig_calculator
        elif eig_method == 'flo':
            from .eig.flo import flo as eig_calculator         
        else:
            raise ValueError(f'Unknown eig method: {eig_method}. Choose from {self.eig_methods}')                                

        start_time = time.perf_counter()

        if random_seed is not None: 
            torch.manual_seed(random_seed)
            #TODO: add documentation for this behaviour
            try:
                self.m_prior_dist._reset_sample_generator()
            except AttributeError:
                pass
        
        out = eig_calculator(self, design, **eig_method_kwargs)
        
        # deal with nan data
        try:
            out[1]['wall_time'] = time.perf_counter() - start_time
        except TypeError:
            pass
    
        return  out
    

class BED_base_explicit(BED_base):
    r"""
    Defines the base class for Bayesian experimental design methods that do not have nuisance parameters. This is the classic case of Bayesian experimental design.
    
    The data likelihood is assumed to be a function of the model parameters and the experimental design only. The data likelihood is function takes two arguments: the model parameters and the experimental design. The model parameters are assumed to be a tensor of shape (n_model_samples, dim_model_parameters). The experimental design is assumed to be a tensor of shape (design_dim). The data likelihood function returns a :class:`torch.distributions.Distribution` object. The data likelihood function must be provided as an argument to the constructor of the class.
    
    The model parameter prior distribution is assumed to be a :class:`torch.distributions.Distribution` object. The model parameter prior distribution needs to return samples of shape (n_model_samples, dim_model_parameters). The model parameter prior distribution is assumed to be provided as an argument to the constructor of the class. Be carefull when using one dimensional :torch.distributions.Distribution` objects such as :torch.distributions.Normal` or :torch.distributions.Uniform`. Use :torch.distributions.Independent` if necessary to make sure that the samples are of shape (n_model_samples, dim_model_parameters).
    """
    def __init__(
        self,
        m_prior_dist: Union[Distribution, Tensor],
        data_likelihood_func: callable):
        
        super().__init__(m_prior_dist)
        
        if not ((inspect.getfullargspec(data_likelihood_func).args == ['self', 'model_samples', 'design'] \
                or inspect.getfullargspec(data_likelihood_func).args == ['model_samples', 'design']) \
            and inspect.getfullargspec(data_likelihood_func).varargs is None \
            and inspect.getfullargspec(data_likelihood_func).varkw is None):
            raise ValueError('Data likelihood function must have the following signature: data_likelihood(model_samples, design)')
        
        self.data_likelihood_func = data_likelihood_func
        self.nuisance_dist = None
        self.implict_data_likelihood_func = False
        
    def get_data_likelihood(
        self,
        design: Tensor,
        n_model_samples: int=1,
        random_seed_model: int=None,
        ) -> Distribution:
        """
        Samples model parameters from the model prior distribution and return the data likelihood at the design and the sampled parameters.
        
        Args:
            design: Tensor describing the experimental design at which the data likelihood is evaluated.
            n_model_samples: Number of model parameter samples to return. Defaults to 1.
            random_seed_model: Random seed to use for sampling from the model parameter prior distribution. Defaults to None.
        
        Returns:
            Returns a distribution of data samples.
        """
        model_samples = self.get_m_prior_samples(n_model_samples, random_seed_model)
        return self.data_likelihood_func(
            model_samples = model_samples,
            design = design), model_samples
        
    def get_data_likelihood_samples(
        self,
        design: Tensor,
        n_model_samples: int=1,
        n_likelihood_samples: int=1,
        random_seed_model: int=None,
        random_seed_likelihood: int=None,
        ) -> Tensor:
        """
        Samples model parameters from their prior distribution, evaluates the data likelihood at the design and the sampled model parameters and returns samples from the data likelihood distribution.
        
        Args:
            design: Tensor describing the experimental design at which the data likelihood is evaluated.
            n_model_samples: Number of model parameter samples to return. Defaults to 1.
            n_likelihood_samples: Number of samples to return from the data likelihood distribution. Defaults to 1.
            random_seed_model: Random seed to use for sampling from the model parameter prior distribution. Defaults to None.
            random_seed_likelihood: Random seed to use for sampling from the data likelihood distribution. Defaults to None.
        
        Returns:
            Returns a tensor of data samples of shape (n_likelihood_samples, \*n_model_samples, dim_data_samples).
        """
        if type(n_likelihood_samples) == int:
            n_likelihood_samples = (n_likelihood_samples,)
            
        data_likelihood, model_samples = self.get_data_likelihood(design, n_model_samples, random_seed_model)
        
        if random_seed_likelihood is not None:
            torch.manual_seed(random_seed_likelihood)
        
        return data_likelihood.sample(n_likelihood_samples).squeeze(0), model_samples

class BED_base_nuisance(BED_base):
    r"""
    Defines the base class for Bayesian experimental design methods that have nuisance parameters. This is the case when the data likelihood is a function of the nuisance parameters, the model parameters, and the experimental design. The data likelihood is function takes three arguments: the experimental design, the model parameters and the nuisance parameters. The model parameters are assumed to be a tensor of shape (n_model_samples, dim_model_parameters). The nuisance parameters are assumed to be a tensor of shape (n_nuisance_samples, \*n_model_samples, dim_nuisance_parameters). The experimental design is assumed to be a tensor of shape (design_dim). The data likelihood function returns a :class:`torch.distributions.Distribution` object. The data likelihood function must be provided as an argument to the constructor of the class.
    
    The model parameter prior distribution is assumed to be a :class:`torch.distributions.Distribution` object. The model parameter prior distribution needs to return samples of shape (n_model_samples, dim_model_parameters). The model parameter prior distribution is assumed to be provided as an argument to the constructor of the class. Be carefull when using one dimensional :torch.distributions.Distribution` objects such as :torch.distributions.Normal` or :torch.distributions.Uniform`. Use :torch.distributions.Independent` if necessary to make sure that the samples are of shape (n_model_samples, dim_model_parameters).
    
    The nuisance parameter prior distribution can be either a :class:`torch.distributions.Distribution` object or a function that takes the model parameters as an argument and returns a :class:`torch.distributions.Distribution` object. The nuisance parameter prior distribution needs to return samples of shape (n_nuisance_samples, \*n_model_samples, dim_nuisance_parameters). If the nuisance parameter distribution is independent of the model parameters, it is recommended to use a :class:`torch.distributions.Distribution` object, since the provided distribution will automatically be expanded to return samples of shape (n_nuisance_samples, \*n_model_samples, dim_nuisance_parameters).
    
    Args:
        m_prior_dist: The prior distribution of the model parameters. Must be a :class:`torch.distributions.Distribution` object.
        nuisance_dist: The prior distribution of the nuisance parameters. Must be a :class:`torch.distributions.Distribution` object or a function that takes the model parameters as an argument and returns a :class:`torch.distributions.Distribution` object. 
        data_likelihood_func: The data likelihood function. Must be a function that takes three arguments: the experimental design, the model parameters and the nuisance parameters. The model parameters are assumed to be a tensor of shape (n_nuisance_samples, \*n_model_samples, dim_model_parameters). The nuisance parameters are assumed to be a tensor of shape (n_nuisance_samples, \*n_model_samples, dim_nuisance_parameters). The experimental design is assumed to be a tensor of shape (design_dim). The data likelihood function returns a :class:`torch.distributions.Distribution` object.
    """ 
    def __init__(
        self,
        m_prior_dist: Union[Distribution, Tensor],
        nuisance_dist: Union[Distribution, callable, Tensor],
        data_likelihood_func: callable,
        ):
        
        super().__init__(m_prior_dist)
        # Enforce that nuisance distribution might be conditional on model parameters
        if not callable(nuisance_dist):
            # Check if nuisance samples or nuisance distribution is provided
            if type(nuisance_dist) == torch.Tensor:
                nuisance_dist = SampleDistribution(nuisance_dist)
            self.nuisance_dist = _Dummy_Cond_Dist(nuisance_dist)
        else:
            self.nuisance_dist = nuisance_dist
                        
        if not ((inspect.getfullargspec(data_likelihood_func).args == ['nuisance_samples', 'model_samples', 'design'] or \
                 inspect.getfullargspec(data_likelihood_func).args == ['self', 'nuisance_samples', 'model_samples', 'design']) \
            and inspect.getfullargspec(data_likelihood_func).varargs is None \
            and inspect.getfullargspec(data_likelihood_func).varkw is None):
            raise ValueError('Data likelihood function must have the following signature: data_likelihood(nuisance_samples, model_samples, design)')
            
        self.data_likelihood_func = data_likelihood_func
        self.independent_nuisance_parameters = False
        self.implict_data_likelihood_func = False
    
    def get_nuisance_samples(
        self,
        model_samples: Tensor,
        n_nuisance_samples: int=1,
        random_seed: int=None,
        ) -> Tensor:
        """
        Samples nuisance parameters from their prior distribution.
        
        Args:
            model_samples: Tensor of model parameter samples of shape (n_model_samples, dim_model_parameters).
            n_nuisance_samples: Number of nuisance parameter samples to return. Defaults to 1.
            random_seed_nuisance: Random seed to use for sampling from the nuisance parameter prior distribution. Defaults to None.
        
        Returns:
            Returns a tensor of nuisance parameter samples of shape (n_nuisance_samples, \*n_model_samples, dim_nuisance_parameters).
        """        
        if type(n_nuisance_samples) == int:
            n_nuisance_samples = (n_nuisance_samples,)
        
        if random_seed is not None:
            torch.manual_seed(random_seed)
                
        return self.nuisance_dist(model_samples).sample(
            n_nuisance_samples).squeeze(0)
        
    
    def get_data_likelihood(
        self,
        design: Tensor,
        n_model_samples: int=1,
        n_nuisance_samples: int=1,
        random_seed_model: int=None,
        random_seed_nuisance: int=None,
        ) -> Distribution:
        """
        Samples model and nuisance parameters from their prior distributions and evaluates the data likelihood at the design and the sampled parameters.
        
        Args:
            design: Tensor describing the experimental design at which the data likelihood is evaluated.
            n_model_samples: Number of model parameter samples to return. Defaults to 1.
            n_nuisance_samples: Number of nuisance parameter samples to return. Defaults to 1.
            random_seed_model: Random seed to use for sampling from the model parameter prior distribution. Defaults to None.
            random_seed_nuisance: Random seed to use for sampling from the nuisance parameter prior distribution. Defaults to None.
        
        Returns:
            Returns a distribution of data samples.
        """        
        model_samples = self.get_m_prior_samples(n_model_samples, random_seed_model)
        
        nuisance_samples = self.get_nuisance_samples(model_samples, n_nuisance_samples, random_seed_nuisance)
        model_samples = model_samples.expand(*nuisance_samples.shape[:-2], -1, -1)
                        
        return self.data_likelihood_func(
            nuisance_samples = nuisance_samples,
            model_samples = model_samples,
            design = design), model_samples
    
    
    def get_data_likelihood_samples(
        self,
        design: Tensor,
        n_model_samples: int=1,
        n_nuisance_samples: int=1,
        n_likelihood_samples: int=1,
        random_seed_model: int=None,
        random_seed_nuisance: int=None,
        random_seed_likelihood: int=None,
        ) -> Tensor:
        """
        Samples model and nuisance parameters from their prior distributions, evaluates the data likelihood at the design and the sampled parameters and returns samples from the data likelihood distribution.
        
        Args:
            design: Tensor describing the experimental design at which the data likelihood is evaluated.
            n_model_samples: Number of model parameter samples to return. Defaults to 1.
            n_nuisance_samples: Number of nuisance parameter samples to return. Defaults to 1.
            n_likelihood_samples: Number of samples to return from the data likelihood distribution. Defaults to 1.
            random_seed_model: Random seed to use for sampling from the model parameter prior distribution. Defaults to None.
            random_seed_nuisance: Random seed to use for sampling from the nuisance parameter prior distribution. Defaults to None.
            random_seed_likelihood: Random seed to use for sampling from the data likelihood distribution. Defaults to None.
        
        Returns:
            Returns a tensor of data samples of shape (n_likelihood_samples, \*n_model_samples, \*n_nuisance_samples, dim_data_samples).
        """
        if type(n_likelihood_samples) == int:
            n_likelihood_samples = (n_likelihood_samples,)
            
        data_likelihood, model_samples = self.get_data_likelihood(
            design, n_model_samples, n_nuisance_samples, random_seed_model, random_seed_nuisance)
        
        if random_seed_likelihood is not None:
            torch.manual_seed(random_seed_likelihood)
        
        return data_likelihood.sample(n_likelihood_samples) , model_samples


# class BED_base_implicit(BED_base):
#     pass