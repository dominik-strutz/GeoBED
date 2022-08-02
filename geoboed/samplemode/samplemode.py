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

class BOED_sample():
    def __init__(self, filename, model_prior=None, disable_tqdm=False, **kwargs):
        """_summary_

        Args:
            filename (_type_): Filname of hdf5 file containing the prior, the design ad the data labeled as such. 
        """
        #TODO: Make more flexible for other filetypes such as npy, csv, etc.
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
        
        if self.filetype == 'hdf5':
            with h5py.File((filename), "r") as hdf:
                self.n_prior = hdf['prior'].shape[0]
                self.n_design = hdf['design'].shape[0]
                self.design_dim = hdf['design'].shape[1] #TODO: Allow for different design dimensions
                
                #answer: design dim is set to largest here and use nan to denote missing entries, needs to be implemented on the levels of iniodividual quality evaluation functions

                assert self.n_prior    == hdf['data'].shape[0], 'The number of prior samples     and size of the second dimension of the data array must be the same.'
                assert self.n_design   == hdf['data'].shape[1], 'The number of design samples    and size of the fist   dimension of the data array must be the same.'
                assert self.design_dim == hdf['data'].shape[2], 'The number of design dimensions and size of the third  dimension of the data array must be the same.'
                
        elif self.filetype == 'npz':
            with open(filename, 'rb') as npz:
                npz_data = np.load(npz)
                
                self.n_prior = npz_data['prior'].shape[0]
                self.n_design = npz_data['design'].shape[0]
                self.design_dim = npz_data['design'].shape[1]
                
                assert self.n_prior    == npz_data['data'].shape[0], 'The number of prior samples     and size of the second dimension of the data array must be the same.'
                assert self.n_design   == npz_data['data'].shape[1], 'The number of design samples    and size of the fist   dimension of the data array must be the same.'
                assert self.design_dim == npz_data['data'].shape[2], 'The number of design dimensions and size of the third  dimension of the data array must be the same.'
                

    def get_design(self):
        
        if self.filetype == 'hdf5':
            with h5py.File((self.filename), "r") as hdf:
                samples = hdf['design'][:]
        elif self.filetype == 'npz':
            with open(self.filename, 'rb') as npz:
                npz_data = np.load(npz)
                samples = npz_data['design'][:]
                    
        return samples
    
    def get_prior(self, n_samples=None):
        
        if self.filetype == 'hdf5':
            with h5py.File((self.filename), "r") as hdf:
                if n_samples is None:
                    samples = hdf['prior'][:]
                else:   
                    samples = hdf['prior'][:n_samples]
        elif self.filetype == 'npz':
            with open(self.filename, 'rb') as npz:
                npz_data = np.load(npz)
                if n_samples is None:
                    samples = npz_data['prior'][:]
                else:   
                    samples = npz_data['prior'][:n_samples]
                    
        return samples
    
    def get_samples(self, n_samples=None):
        
        if self.filetype == 'hdf5':
            with h5py.File((self.filename), "r") as hdf:
                if n_samples is None:
                    samples = hdf['data'][:]
                else:   
                    samples = hdf['data'][:, :n_samples, :]
        elif self.filetype == 'npz':
            with open(self.filename, 'rb') as npz:
                npz_data = np.load(npz)
                if n_samples is None:
                    samples = npz_data['data'][:]
                else:   
                    samples = npz_data['data'][:, :n_samples, :]
                    
        return samples
    
    def get_noisy_samples(self, n_samples=None):
        """_summary_

        Args:
            n_samples (_type_): _description_
        """
        if n_samples is None:
            n_samples = self.n_prior
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
    
    def find_optimal_design(self, boed_method, boed_method_kwargs={}, **kwargs):
        """_summary_

        Args:
            boed_method (_type_): _description_
        """        
        if boed_method in ['dn', 'nmc', 'var_marg', 'var_post']:
            self.boed_method = boed_method
        else:
            raise NotImplementedError('BOED method not implemented. Please choose from one of the following: dn, nmc, var_marg, var_post')
    
        if self.filetype == 'hdf5':

            with h5py.File((self.filename), "r") as hdf:

                if self.boed_method == 'dn':
                    self.oed_results = dn(self, hdf, **boed_method_kwargs)
                elif self.boed_method == 'nmc':
                    self.oed_results = nmc(self, hdf, **boed_method_kwargs)
                elif self.boed_method == 'var_marg':
                    self.oed_results = var_marg(self, hdf, **boed_method_kwargs)
                elif self.boed_method == 'var_post':
                    self.oed_results = var_post(self, hdf, **boed_method_kwargs)
                            
        elif self.filetype == 'npz':
             with open(self.filename, 'rb') as npz:
                npz_data = np.load(npz)

                if self.boed_method == 'dn':
                    self.oed_results = dn(self, npz_data, **boed_method_kwargs)
                elif self.boed_method == 'nmc':
                    self.oed_results = nmc(self, npz_data, **boed_method_kwargs)
                elif self.boed_method == 'var_marg':
                    self.oed_results = var_marg(self, npz_data, **boed_method_kwargs)
                elif self.boed_method == 'var_post':
                    self.oed_results = var_post(self, npz_data, **boed_method_kwargs)
            
        return self.oed_results

            
            

        


            
        
            



        
        
    