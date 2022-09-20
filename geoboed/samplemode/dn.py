import math

from tqdm import tqdm
import torch
import pyro
import numpy as np

def dn(self, dataframe, design_list, N=-1, return_dict=False, preload_samples=True, disable_tqdm=False, set_rseed=True):
    """_summary_

    Args:
        **kwargs: _description_
    """
    
    N = N if not N==-1 else self.n_prior
    
    #TODO: Implement determinant of std errors for design dependent gaussian
    #TODO: Implement test for gaussian noise and design independent noise
    
    eig_list = []
    
    if preload_samples:
        preloaded_samples = torch.tensor(dataframe['data'][:N])    
    
    for i, design_i in tqdm(enumerate(design_list), total=len(design_list), disable=disable_tqdm):
        
        if preload_samples:
            samples = preloaded_samples[:, design_i]
        else:
            samples = torch.tensor(dataframe['data'][:N, design_i])
                
        if set_rseed:
            pyro.set_rng_seed(0)
            torch.manual_seed(0)
        
        samples = self.data_likelihood(samples, design_i).sample([1, ]).flatten(start_dim=0, end_dim=1)
                        
        # determinant of 1D array covariance not possible so we need to differentiate
        if len(design_i) < 2:
            model_det = torch.cov(samples.T).detach()
        else:
            model_det = torch.linalg.det(torch.cov(samples.T))
        #TODO: Test if this really works                 
        D = design_i.shape[0] 
                    
        eig = 1/2 * math.log(model_det) + D/1 + D/1 * math.log(2*math.pi)
        
        eig_list.append(eig)        
    
    if return_dict:
        output_dict = {'N': N,}
        return np.array(eig_list), output_dict
        
    else:
        return np.array(eig_list), None
