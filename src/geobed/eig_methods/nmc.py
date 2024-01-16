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

    if random_seed is not None:
        torch.manual_seed(random_seed)

    if self.implict_data_likelihood_dist:
        raise ValueError("NMC cannot be used with implicit observation noise distribution")

    if M is None and not reuse_M:
        raise ValueError("M must be provided if reuse_M is False")
    
    if reuse_M:
        if M is not None:
            raise ValueError("M must not be provided if reuse_M is True")
        M = N

    if self.nuisance_dist is None:
        if M_prime is not None:
            raise ValueError("M_prime not needed for NMC without nuisance parameters")
        
        N_data_likelihoods, _ = self.get_data_likelihood(
                design, n_model_samples=N)
        
        if reuse_M:
            if not memory_efficient:
                M_data_likelihoods = N_data_likelihoods.expand((M, N))
            else:
                M_data_likelihoods = N_data_likelihoods
        else:
            M_data_likelihoods, _ = self.get_data_likelihood(
                design, n_model_samples=(M, N))
            
    else:
        if M_prime is None:
            raise ValueError("M_prime must be provided for NMC with nuisance parameters")
        
        N_data_likelihoods, _ = self.get_data_likelihood(
            design, n_model_samples=N, n_nuisance_samples=M_prime)
        
        M_data_likelihoods, _ = self.get_data_likelihood(
                    design, n_model_samples=N, n_nuisance_samples=1)
        
        if reuse_M:
            if not memory_efficient:
                M_data_likelihoods = M_data_likelihoods.expand((M, N))
            
        else:
            raise ValueError("Nuisance parameters not implemented for NMC without reuse_M")

    # print('N_data_likelihoods', N_data_likelihoods)
    # print('M_data_likelihoods', M_data_likelihoods)
    
    if self.nuisance_dist is None:
        N_data_samples = N_data_likelihoods.sample()
        M_data_samples = N_data_samples
    else:
        M_data_samples = M_data_likelihoods.sample().squeeze(0)
        N_data_samples = N_data_likelihoods.sample().squeeze(0)[0]
    
    
    # print('N_data_samples.shape', N_data_samples.shape)
    # print('M_data_samples.shape', M_data_samples.shape)
    
    if not memory_efficient:
        if not reuse_M:
            marginal_lp = M_data_likelihoods.log_prob(M_data_samples).logsumexp(0) - math.log(M)
        else:
            marginal_lp = M_data_likelihoods.log_prob(
                M_data_samples.unsqueeze(0).swapaxes(0, 1)
                ).swapaxes(0, 1).logsumexp(0) - math.log(M)
    if memory_efficient:
        if reuse_M:
            marginal_lp = torch.zeros(N)            
            for i in range(N):
                marginal_lp[i] = M_data_likelihoods.log_prob(M_data_samples[i]).logsumexp(0) - math.log(M)
        
        else:
            raise ValueError("Memory efficient NMC not implemented for non-reuse_M")
        
    #     if reuse_M:
    #         marginal_lp = torch.zeros(N)
    #         for i in range(N):
    #             marginal_lp[i] = self.obs_noise_dist(M_fwd_out, desig n).log_prob(N_samples[i]).logsumexp(0) - math.log(M)
    #     else:
    #         raise ValueError("Memory efficient NMC not implemented for non-reuse_M")
    

    if M_prime is None:
        conditional_lp = N_data_likelihoods.log_prob(N_data_samples)
    else:        
        conditional_lp = (N_data_likelihoods.log_prob(N_data_samples)[1:].logsumexp(0) - math.log(M_prime-1))


    # print('conditional_lp', conditional_lp.sum(0))
    # print('marginal_lp', marginal_lp.sum(0))

    eig = (conditional_lp - marginal_lp).sum(0) / N
        
    out_dict = {'N': N, 'M': M, 'M_prime':M_prime, 'reuse_N_samples': reuse_M, 'memory_efficient': memory_efficient}
    
    # ugag
    
    return eig, out_dict
    
    

    
    # N_likelihoods = self.obs_noise_dist(N_likelihoods_input, design)
    # N_samples = N_likelihoods.sample()
                   
    # if reuse_M:
    #     M_fwd_out = N_fwd_out
    # else:
    #     M_fwd_out = self._get_forward_function_samples(
    #         design, n_samples_model=n_M_samples, n_samples_nuisance=1,)
    
    # if N_fwd_out['data'] == None or M_fwd_out['data'] == None :
    #     return torch.tensor(torch.nan), None

    # if not memory_efficient:
            
    #     NM_likelihoods_input = {}
    #     for key in N_fwd_out:
    #         if key == 'nuisance_samples' and self.nuisance_dist is None:
    #             continue
            
    #         if M_fwd_out[key].ndim == 3: 
    #             data = M_fwd_out[key]
    #         else:
    #             data = M_fwd_out[key].unsqueeze(1)
            
    #         if reuse_M:
    #             NM_likelihoods_input[key] = data.expand(M, N, data.shape[-1])
    #         else:
    #             NM_likelihoods_input[key] = data.reshape(M, N, data.shape[-1])

    #     NM_likelihoods = self.obs_noise_dist(NM_likelihoods_input, design)
        
    #     marginal_lp = NM_likelihoods.log_prob(N_samples).logsumexp(0) - math.log(M)

    # else:
    #     if reuse_M:
    #         marginal_lp = torch.zeros(N)
    #         for i in range(N):
    #             marginal_lp[i] = self.obs_noise_dist(M_fwd_out, design).log_prob(N_samples[i]).logsumexp(0) - math.log(M)
    #     else:
    #         raise ValueError("Memory efficient NMC not implemented for non-reuse_M")
    
    # if M_prime is None:
    #     conditional_lp = N_likelihoods.log_prob(N_samples)
    # else:
        
    #     NM_prime_likelihood_inputs = {}
    #     for key in N_fwd_out:
    #         if key == 'nuisance_samples' and self.nuisance_dist is None:
    #             continue
            
    #         if M_fwd_out[key].ndim == 3: 
    #             data = M_fwd_out[key]
    #         else:
    #             data = M_fwd_out[key].unsqueeze(1)
                
    #         if self.independent_nuisance_parameters:
    #             NM_prime_likelihood_inputs = data.swapaxes(0, 1).reshape(N, M_prime, data.shape[-1])
    #         else:            
    #             NM_prime_likelihood_inputs = data.swapaxes(0, 1)        
                
    #     NM_primelikelihoods = self.obs_noise_dist(NM_prime_likelihood_inputs, design)
    #     conditional_lp = NM_primelikelihoods.log_prob(N_samples.unsqueeze(0)).logsumexp(0) - math.log(M_prime)                

    # eig = (conditional_lp - marginal_lp).sum(0) / N
        
    # out_dict = {'N': N, 'M': M, 'M_prime':M_prime, 'reuse_N_samples': reuse_M, 'memory_efficient': memory_efficient}
        
    # return eig, out_dict

