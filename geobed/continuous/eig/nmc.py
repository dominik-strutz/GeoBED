import math
import torch
import numpy as np

from torch import Tensor

def nmc(
    self,
    design: Tensor,
    N: int,
    M: int = None,
    reuse_M: bool=False,
    memory_efficient: bool=False,
    M_prime: int=None,
    random_seed=None,
    ):

    #TODO: add support for independet priors 

    if random_seed is not None:
        torch.manual_seed(random_seed)

    if self.target_forward_function:
        raise ValueError("NMC cannot be used with target forward mapping")
    if self.implict_obs_noise_dist:
        raise ValueError("NMC cannot be used with implicit observation noise distribution")
    
    if M is not None and reuse_M:
        raise ValueError("M must be provided if reuse_M is False")
    
    if reuse_M:
        n_M_samples = N
        M = N
    else:
        n_M_samples = N * M
            
        
    if self.nuisance_dist is None:
        if M_prime is not None:
            raise ValueError("M_prime not needed for NMC without nuisance parameters")
        N_fwd_out = self._get_forward_function_samples(
            design, n_samples_model=N, n_samples_nuisance=1,)
    else:
        if M_prime is None:
            raise ValueError("M_prime must be provided for NMC with nuisance parameters")
        if self.independent_nuisance_parameters:
            N_fwd_out = self._get_forward_function_samples(
            design, n_samples_model=N, n_samples_nuisance=1,)
        else:
            N_fwd_out = self._get_forward_function_samples(
            design, n_samples_model=N, n_samples_nuisance=M_prime,)
    
    N_likelihoods_input = {}
    for key in N_fwd_out:
        if key == 'nuisance_samples' and self.nuisance_dist is None:
            continue
        
        if N_fwd_out[key].ndim == 3:
            N_likelihoods_input[key] = N_fwd_out[key][:, 0]
        else:
            N_likelihoods_input[key] = N_fwd_out[key]
    
    N_likelihoods = self.obs_noise_dist(N_likelihoods_input, design)
    N_samples = N_likelihoods.sample()
                   
    if reuse_M:
        M_fwd_out = N_fwd_out
    else:
        M_fwd_out = self._get_forward_function_samples(
            design, n_samples_model=n_M_samples, n_samples_nuisance=1,)
    
    if N_fwd_out['data'] == None or M_fwd_out['data'] == None :
        return torch.tensor(torch.nan), None

    if not memory_efficient:
            
        NM_likelihoods_input = {}
        for key in N_fwd_out:
            if key == 'nuisance_samples' and self.nuisance_dist is None:
                continue
            
            if M_fwd_out[key].ndim == 3: 
                data = M_fwd_out[key]
            else:
                data = M_fwd_out[key].unsqueeze(1)
            
            if reuse_M:
                NM_likelihoods_input[key] = data.expand(M, N, data.shape[-1])
            else:
                NM_likelihoods_input[key] = data.reshape(M, N, data.shape[-1])

        NM_likelihoods = self.obs_noise_dist(NM_likelihoods_input, design)
        
        marginal_lp = NM_likelihoods.log_prob(N_samples).logsumexp(0) - math.log(M)

    else:
        if reuse_M:
            marginal_lp = torch.zeros(N)
            for i in range(N):
                marginal_lp[i] = self.obs_noise_dist(M_fwd_out, design).log_prob(N_samples[i]).logsumexp(0) - math.log(M)
        else:
            raise ValueError("Memory efficient NMC not implemented for non-reuse_M")
    
    if M_prime is None:
        conditional_lp = N_likelihoods.log_prob(N_samples)
    else:
        
        NM_prime_likelihood_inputs = {}
        for key in N_fwd_out:
            if key == 'nuisance_samples' and self.nuisance_dist is None:
                continue
            
            if M_fwd_out[key].ndim == 3: 
                data = M_fwd_out[key]
            else:
                data = M_fwd_out[key].unsqueeze(1)
                
            if self.independent_nuisance_parameters:
                NM_prime_likelihood_inputs = data.swapaxes(0, 1).reshape(N, M_prime, data.shape[-1])
            else:            
                NM_prime_likelihood_inputs = data.swapaxes(0, 1)        
                
        NM_primelikelihoods = self.obs_noise_dist(NM_prime_likelihood_inputs, design)
        conditional_lp = NM_primelikelihoods.log_prob(N_samples.unsqueeze(0)).logsumexp(0) - math.log(M_prime)                

    eig = (conditional_lp - marginal_lp).sum(0) / N
        
    out_dict = {'N': N, 'M': M, 'M_prime':M_prime, 'reuse_N_samples': reuse_M, 'memory_efficient': memory_efficient}
        
    return eig, out_dict

