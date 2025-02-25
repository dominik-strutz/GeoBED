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
    M_prime:int=None,
    random_seed:int=None,
    ):
    r"""
    Calculate the expected information gain (EIG) using the :math:`D_N` method.
    """

    if random_seed is not None:
        torch.manual_seed(random_seed)

    if self.implict_data_likelihood_func:
        raise ValueError(r"$D_N$ method cannot be used with implicit observation noise distribution")
    
    if self.nuisance_dist is not None:
        if M_prime is None:
            raise ValueError("M_prime must be provided for DN with nuisance parameters")
        data_likelihoods, _ = self.get_data_likelihood(
            design, n_model_samples=N, n_nuisance_samples=M_prime)
        data_samples = data_likelihoods.sample()[0]
        conditional_lp = data_likelihoods.log_prob(data_samples).logsumexp(0) - math.log(M_prime)
    else:
        if M_prime is not None:
            raise ValueError("M_prime not needed for DN without nuisance parameters")
        data_likelihoods, _ = self.get_data_likelihood(
            design, n_model_samples=N)
        data_samples = data_likelihoods.sample()
        conditional_lp = data_likelihoods.log_prob(data_samples).detach()

    D_dim = data_samples.shape[-1]
    # determinant of 1D array covariance not possible 
    if D_dim < 2:
        model_det = math.log(torch.cov(data_samples.T))
    else:
        sig_det, val_det = torch.slogdet(torch.cov(data_samples.T))
        model_det = sig_det * val_det

    marginal_lp = -1/2 * (model_det + D_dim + D_dim * math.log(2*math.pi))

    # eig = ((conditional_lp).sum(0) / N) - marginal_lp
    eig = conditional_lp - marginal_lp
    eig = eig.nansum(0) / torch.isfinite(eig).sum(0)

    out_dict = {'N': N}

    return eig, out_dict