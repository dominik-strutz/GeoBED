r"""
Calculate the expected information gain (EIG) using the :math:`D_N` method.
"""

import math
import torch
import numpy as np

from torch import Tensor

def dn(
    self,
    design: Tensor,
    N:int,
    random_seed:int=None,
    ):
    r"""
    Calculate the expected information gain (EIG) using the :math:`D_N` method.
    """

    if (self.nuisance_dist is not None) \
        or self.target_forward_function \
            or self.implict_obs_noise_dist:
        raise ValueError(r"$D_N$ method cannot be used with implicit likelihoods.") 

    if random_seed is not None:
        torch.manual_seed(random_seed)
        
    data_likelihoods = self.get_foward_model_distribution(design, N)
    data_samples = data_likelihoods.sample()

    D_dim = data_samples.shape[-1]
    # determinant of 1D array covariance not possible 
    if D_dim < 2:
        model_det = math.log(torch.cov(data_samples.T))
    else:
        sig_det, val_det = torch.slogdet(torch.cov(data_samples.T))
        model_det = sig_det * val_det

    marginal_lp = -1/2 * (model_det + D_dim + D_dim * math.log(2*math.pi))
    conditional_lp = data_likelihoods.log_prob(data_samples).detach()

    eig = ((conditional_lp).sum(0) / N) - marginal_lp
    
    out_dict = {'N': N}

    return eig, out_dict