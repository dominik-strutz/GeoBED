import math

import numpy as np
import h5py
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from .dn import *
from .nmc import *
from .var_marg import *
from .var_post import *
from .dataloader import *


def parallel_helper():
    print('helper')

class BOED_sample():
    def __init__(self, filename, model_prior=None, disable_tqdm=False, **kwargs):
        """_summary_

        Args:
            filename (_type_): Filname of hdf5 file containing the prior, the design ad the data labeled as such. 
        """
        #TODO: Implement generator mode
        
        self.filename = filename
        self.disable_tqdm = disable_tqdm 

        #TODO: implement as seperate method like likeliehood
        #Dont sample from this!!!! 
        self.model_prior = model_prior
        
        if filename.endswith('.hdf5'):
            self.filetype = 'hdf5'
        elif filename.endswith('.npz'):
            self.filetype = 'npz'
        else:
            raise ValueError('Filetype not supported')
        
        with dataloader(filename) as dataframe:
            self.n_prior    = dataframe['prior'].shape[0]
            self.model_dim    = dataframe['prior'].shape[1]
            self.n_design   = dataframe['design'].shape[0]
            self.design_dim = dataframe['design'].shape[1]

            assert self.n_prior    == dataframe['data'].shape[0], 'The number of prior samples     and size of the second dimension of the data array must be the same.'
            assert self.n_design   == dataframe['data'].shape[1], 'The number of design samples    and size of the fist   dimension of the data array must be the same.'

    def get_designs(self):
        
        with dataloader(self.filename) as dataframe:
            samples = dataframe['design'][:]
                    
        return samples
    
    def get_prior(self, n_samples=None):
        
        n_samples = self.n_prior if n_samples is None else n_samples
        if n_samples > self.n_prior:
            raise ValueError('Only {self.n_prior} samples are available!')
        
        with dataloader(self.filename) as dataframe:
            samples = dataframe['prior'][:n_samples]
                    
        return samples
    
    def get_samples(self, n_samples=None):
        
        n_samples = self.n_prior if n_samples is None else n_samples
        if n_samples > self.n_prior:
            raise ValueError('Only {self.n_prior} samples are available!')
        
        with dataloader(self.filename) as dataframe:
            samples = dataframe['data'][:n_samples]
                    
        return samples
                        
    def get_noisy_samples(self, n_samples=None):
        """_summary_

        Args:
            n_samples (_type_): _description_
        """
        samples = torch.tensor(self.get_samples(n_samples))
        samples = self.data_likelihood(samples).sample([1, ]).flatten(start_dim=0, end_dim=1)
        return samples.numpy()

    def set_data_likelihood(self, data_likelihood, design_independent=False, **kwargs):
        """_summary_

        Args:
            data_likelihood (_type_): _description_
        """
        self.data_likelihood = data_likelihood
        self.design_independent_likelihood = design_independent
    
    def find_optimal_design(self, n_design, optimization_method, 
                            boed_method, boed_method_kwargs={},
                            return_dict=True, design_restriction=None, **kwargs):
        """_summary_

        Args:
            boed_method (_type_): _description_
        """        

        if optimization_method == 'iterative construction':
            raise NotImplementedError('Iterative construction not implemented yet')
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


        
        #TODO: do setting of design here
        #TODO: do data loading here, especially with regards to multidimensional designs each method should return (eig_list, process information) process information would be a dictionary of {loss, guide, information like (batched, N, M, stochastic or not, ...) }

        # return self.oed_results
    
    
    def get_eig(self, design_list, boed_method, boed_method_kwargs={},
                return_dict=True,
                **kwargs):
        
        if boed_method in ['dn', 'nmc', 'var_marg', 'var_post']:
            self.boed_method = boed_method
        else:
            raise ValueError('BOED method not implemented. Please choose from one of the following: dn, nmc, var_marg, var_post')
        
        with dataloader(self.filename) as dataframe:
            if self.boed_method == 'dn':
                self.oed_results = dn(self, dataframe, design_list, **boed_method_kwargs)
            elif self.boed_method == 'nmc':
                self.oed_results = nmc(self, dataframe, design_list, **boed_method_kwargs)
            elif self.boed_method == 'var_marg':
                self.oed_results = var_marg(self, dataframe, design_list, **boed_method_kwargs)
            elif self.boed_method == 'var_post':
                self.oed_results = var_post(self, dataframe, design_list, **boed_method_kwargs)
                            
        return self.oed_results
        
        
        
