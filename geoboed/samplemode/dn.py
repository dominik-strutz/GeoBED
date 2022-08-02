import math

from tqdm import tqdm
import torch
import pyro
import numpy as np

def dn(self, hdf, N=-1, design_selection=None, **kwargs):
    """_summary_

    Args:
        **kwargs: _description_
    """
    
    N = N if not N==-1 else self.n_prior
    
    #TODO: Implement determinant of std errors for design dependent gaussian
    #TODO: Implement test for gaussian noise and design independent noise
    
    eig_list = []
    
    
    for i, design_i in tqdm(enumerate(hdf['design']), total=self.n_design, disable=self.disable_tqdm):
        
        if design_selection is not None:
            if i not in design_selection:
                continue
        
        pyro.set_rng_seed(0)
        
        samples = torch.tensor(hdf['data'][:N, i])
        samples = self.data_likelihood(samples).sample([1, ]).flatten(start_dim=0, end_dim=1)
                
        # determinant of 1D array covariance not possible so we need to differentiate
        if self.design_dim < 2:
            model_det = torch.cov(samples.T).detach().item()
        else:
            model_det = torch.linalg.det(torch.cov(samples.T)).detach().item()
                                    
        D = self.design_dim 
                    
        eig = 1/2 * math.log(model_det) + D/1 + D/1 * math.log(2*math.pi)
        
        eig_list.append(eig)
        # print(eig)
        
    return np.array(eig_list)