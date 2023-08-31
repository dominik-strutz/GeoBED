import math
import torch
import numpy as np

from torch import Tensor


def laplace(
    self,
    design: Tensor,
    N:int,
    laplace_type:str = 'model',
    random_seed:int=None,
    ):
    
    if (self.nuisance_dist is not None) \
        or self.target_forward_function \
            or self.implict_obs_noise_dist:
        raise ValueError("Laplace method cannot be used with implicit likelihoods.") 
    
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    model_samples = self.get_m_prior_samples(N)
    data_samples
    
    
    print(model_samples.shape)
    
    # def 
    
        
    if laplace_type == 'model':
        

        for m_i in model_samples:
            pass
            
            
            
            
            

        print(jacobians.shape)
        print(hessians.shape)
        
        H_F = hessians.swa




    else:
        raise ValueError("Laplace type not recognised.")