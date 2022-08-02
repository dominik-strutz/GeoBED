import math
from turtle import shape

from tqdm import tqdm
import torch
import pyro
import numpy as np

def nmc(self, hdf, N, M, reuse_N=False, design_selection=None, **kwargs):
        """_summary_

        Args:
            **kwargs: _description_
        """
        #TODO: Implement k-d tree for second estimate if likelihood is deisgn indepenmdet 
        # TODO: Implement logprob batches proberly
        logprob_batches = 2
        
        N = N if not N==-1 else self.n_prior
        M = M if not M==-1 else N
        
        if not reuse_N:
            if N * M > self.n_prior: raise ValueError('Not enough prior samples, choose different N and M!')
        else:
            if N > self.n_prior: raise ValueError('Not enough prior samples, choose different N!')
            if M > self.n_prior: raise ValueError('Not enough prior samples, choose different M!')

        if N < M: raise ValueError("M can't be bigger than N!")

                
        eig_list = []
        
        for i, design_i in tqdm(
            enumerate(hdf['design']), total=self.n_design, disable=self.disable_tqdm):
            
            if design_selection is not None:
                if i not in design_selection:
                    continue
        
            pyro.set_rng_seed(0)

            samples = torch.tensor(hdf['data'][:N, i])
            likelihoods = self.data_likelihood(samples, design_i)
                        
            if reuse_N:
                N_samples  = likelihoods.sample([1, ]).flatten(start_dim=0, end_dim=1)
                NM_samples = likelihoods.sample([M, ]).swapaxes(0, 1)

            else:
                NM_samples = torch.tensor(hdf['data'][:N*M, i])
                NM_likelihoods = self.data_likelihood(NM_samples, design_i)
                NM_samples = NM_likelihoods.sample([1, ]).reshape(M, N, self.design_dim)
                N_samples = NM_samples[0]

            NM_likelihoods = self.data_likelihood(NM_samples, design_i)
            marginal_lp = NM_likelihoods.log_prob(N_samples).logsumexp(0) - math.log(M)
                        
            if self.design_independent_likelihood:
                eig = (-marginal_lp).sum(0) / N
            else:
                conditional_lp = likelihoods.log_prob(N_samples)
                eig = (conditional_lp - marginal_lp).sum(0) / N

            # print(eig.shape)

            eig_list.append(eig.detach().item())
        
        return np.array(eig_list)