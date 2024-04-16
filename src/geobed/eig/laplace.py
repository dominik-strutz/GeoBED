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
    save_IG=False,
    ):
            
    if self.implict_data_likelihood_func:
        raise ValueError("Laplace method cannot be used with implicit likelihoods.")
    
    if random_seed is not None:
        torch.manual_seed(random_seed)
    
    model_samples_ = self.get_m_prior_samples(N)
    d = model_samples_.shape[-1]
    
    model_jacobian = None
    if self.nuisance_dist is not None:
        nuisance_samples = self.get_nuisance_samples(model_samples_, 1).squeeze(0)

        if hasattr(self.data_likelihood_dist, 'model_jacobian'):
            model_jacobian = self.data_likelihood_dist.model_jacobian(
            model_samples_, nuisance_samples, design=design)

        model_samples = zip(
            model_samples_.unsqueeze(1), nuisance_samples.unsqueeze(1))
        
        covariance_matrices = self.data_likelihood_dist(
            model_samples_,
            nuisance_samples.unsqueeze(0),
            design).covariance_matrix.reshape(
                model_samples_.shape[0],
                design.shape[0], design.shape[0])
    
    else:
        if hasattr(self.data_likelihood_dist, 'model_jacobian'):
                        
            model_jacobian = self.data_likelihood_dist.model_jacobian(
            model_samples_, design=design)
        
        model_samples = model_samples_.unsqueeze(1)
        
        covariance_matrices = self.data_likelihood_dist(
        model_samples_,
        design).covariance_matrix.reshape(
            model_samples_.shape[0],
            design.shape[0], design.shape[0])
    
    IG = torch.zeros(N)

    if laplace_type == 'model':
        
        for i, m_i in enumerate(model_samples):
            # try:            
            if not hasattr(self.m_prior_dist, 'log_probs'):
                prior_log_prob_i = self.m_prior_dist.log_prob(m_i[0])
            else:
                prior_log_prob_i = self.m_prior_dist.log_probs[i]

            if not hasattr(self.m_prior_dist, 'hessians'):
                prior_H = torch.autograd.functional.hessian(
                    self.m_prior_dist.log_prob, m_i[0]).squeeze()
            else:
                prior_H = self.m_prior_dist.hessians[i]
            
            if model_jacobian is None:
                raise ValueError("Model Jacobian not found.")
            else:
                fwd_jacobian = model_jacobian[i]
            
            data_cov = covariance_matrices[i].double()

            if torch.allclose(data_cov, torch.diag(data_cov)):
                data_cov_inv = torch.diag(1.0 / torch.diag(data_cov))
            else:
                data_cov_inv = torch.inverse(data_cov)

            # convert to double precision to avoid numerical issues
            Sigma_inv = fwd_jacobian.T.double() @ data_cov_inv @ fwd_jacobian.double() - prior_H.double() + 1e-9 * torch.eye(d)
            Sigma = torch.inverse(Sigma_inv)

            IG[i] = -0.5 * torch.logdet(Sigma) - d/2 * (math.log(2 * torch.pi) + 1) - prior_log_prob_i.double()
        
            
            # except:
            #     IG[i] = torch.tensor(np.nan)

    else:
        raise ValueError("Laplace type not recognised.")
    
    IG = IG.detach()
    
    eig = IG.nanmean().detach()
    
    if save_IG:
        out_dict = {'N': N, 'IG': IG, 'laplace_type': laplace_type,}
    else:
        out_dict = {'N': N, 'laplace_type': laplace_type,}
        
    return eig, out_dict