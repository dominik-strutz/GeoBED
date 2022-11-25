import math
from tqdm import tqdm
import torch
import pyro
import numpy as np
import warnings



def nmc(self, dataframe, design_list, N, M, reuse_N=False, evidence_only=False,
        preload_samples=True, set_rseed=True, disable_tqdm=False):
    """_summary_
    Args:
        **kwargs: _description_
    """
    #TODO: Implement k-d tree for second estimate if likelihood is deisgn indepenmdet 
    # TODO: Implement logprob batches proberly
        
    N = N if not N==-1 else self.n_prior
    M = M if not M==-1 else N
    
    if not reuse_N:
        if N * M > self.n_prior: raise ValueError('Not enough prior samples, choose different N and M!')
    else:
        if N > self.n_prior: raise ValueError('Not enough prior samples, choose different N!')
        if M > self.n_prior: raise ValueError('Not enough prior samples, choose different M!')

    if N < M: raise ValueError("M can't be bigger than N!")

    eig_list = []
    
    if preload_samples:
        if reuse_N:
            pre_samples = torch.tensor(np.apply_along_axis(self.design_restriction, 1, dataframe['data'][:N]))
        else:
            pre_samples = torch.tensor(np.apply_along_axis(self.design_restriction, 1, dataframe['data'][:N*M]))
    
    for i, design_i in tqdm(enumerate(design_list), total=len(design_list), disable=disable_tqdm):
        
        if set_rseed:
            pyro.set_rng_seed(0)
            torch.manual_seed(0)     
           
        if preload_samples:
            samples = pre_samples[:N, design_i]
        else:
            samples = torch.tensor(np.apply_along_axis(self.design_restriction, 1, dataframe['data'][:N])[design_i])

        likelihoods = self.data_likelihood(samples, self.get_designs()[design_i])
                
        if reuse_N:
            N_samples  = likelihoods.sample([1, ]).flatten(start_dim=0, end_dim=1)
            NM_samples = likelihoods.sample([M, ]).swapaxes(0, 1).reshape(M, N, len(design_i))
        else:
            if preload_samples:
                NM_samples = pre_samples[:N*M, design_i]
            else:
                NM_samples = torch.tensor(np.apply_along_axis(self.design_restriction, 1, dataframe['data'][:N*M])[design_i])
            NM_likelihoods = self.data_likelihood(NM_samples, self.get_designs()[design_i])
            NM_samples = NM_likelihoods.sample([1, ]).reshape(M, N, len(design_i))
            N_samples = NM_samples[0]

        NM_likelihoods = self.data_likelihood(NM_samples, self.get_designs()[design_i])
        marginal_lp = NM_likelihoods.log_prob(N_samples).logsumexp(0) - math.log(M)

        if evidence_only:
            if self.design_independent_likelihood == False:
                warnings.warn("design_independent_likelihood set to False. Only use evidence_only option only if likelihood is design independent!")
            eig = (-marginal_lp).sum(0) / N
        else:
            conditional_lp = likelihoods.log_prob(N_samples)
            eig = (conditional_lp - marginal_lp).sum(0) / N 
        
        if eig is not None:
            eig_list.append(eig.item())

    output_dict = {'N': N, 'M': M, 'reuse_N': reuse_N}
    return np.array(eig_list), output_dict

