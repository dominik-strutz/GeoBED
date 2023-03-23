import math
import torch

from tqdm.autonotebook import tqdm

def nmc(self, design, N, M=-1, reuse_M_samples=False,
        M_prime=None, independent_priors=False, worker_id=None):
    
    N = N if not N==-1 else self.n_prior
    M = M if not M==-1 else N
    
    if N < M: print("M should not be bigger than N! Asymptotically, M is idealy set to sqrt(N).")
    
    total_samples = N * (M+1) if not reuse_M_samples else N+M
    if M_prime is not None:
        if independent_priors:
            total_samples += M_prime
        else:
            total_samples += (M_prime+1) * N
    
    data_samples = self.get_forward_samples(design, total_samples)

    if data_samples == None:
        return torch.nan, None

    d_dicts = {n: self.design_dicts[n] for n in design}
    likelihood_kwargs = {'design': design, 'd_dicts': d_dicts}
    
    if M_prime is None:
        N_likelihoods = self.data_likelihood(data_samples[:N], **likelihood_kwargs)
        N_samples = N_likelihoods.sample()
        conditional_lp = N_likelihoods.log_prob(N_samples)
        
        taken_samples = N
    # else:
    #     if independent_priors:
    #         NM_prime_samples = data_samples[N:N+M_prime].reshape(N+M_prime, 1, data_samples.shape[-1])
    #         NM_prime_likelihoods = self.data_likelihood(NM_prime_samples, **likelihood_kwargs)
            
    #         NM_likelihoods.log_prob(NM_prime_samples).logsumexp(0) - math.log(M_prime)
            
    #         taken_samples = N+M_prime
        
    #     else:



    if reuse_M_samples:        
        #TODO: Check if this is correct
        NM_samples = data_samples[taken_samples:taken_samples+M, None]        
        NM_samples = NM_samples.expand(M, N, data_samples.shape[-1])
    else:
        NM_samples = data_samples[taken_samples:].reshape(M, N, data_samples.shape[-1])
                
    NM_likelihoods = self.data_likelihood(NM_samples, **likelihood_kwargs)
    marginal_lp = NM_likelihoods.log_prob(N_samples).logsumexp(0) - math.log(M)
    
    eig = (conditional_lp - marginal_lp).sum(0) / N
    
    out_dict = {'N': N, 'M': M, 'reuse_N_samples': reuse_M_samples}
    
    return eig, out_dict


def dn(self, design, N=None, worker_id=None):
                
    #TODO: Implement determinant of std errors for design dependent gaussian
    #TODO: Implement test for gaussian noise and design independent noise
    
    N = N if not N == None else self.n_prior
    
    data_samples = self.get_forward_samples(design, N)
    
    if data_samples == None:
        return torch.nan, None

    d_dicts = {n: self.design_dicts[n] for n in design}
    likelihood_kwargs = {'design': design, 'd_dicts': d_dicts}
    
    data_likelihoods = self.data_likelihood(data_samples, **likelihood_kwargs)
    
    if data_likelihoods == None:
        raise ValueError('Likelihood not defined for this design. EIG an not be calculated with the DN method')
    
    data = data_likelihoods.sample()
    
    if data == None:
        return torch.nan, None
    
    # determinant of 1D array covariance not possible so we need to differentiate
                
    D = data.shape[-1] 

    if D < 2:
        model_det = math.log(torch.cov(data.T).detach())
    else:
        sig_det, val_det = torch.slogdet(torch.cov(data.T))
        model_det = sig_det * val_det

    marginal_lp = -1/2 * (model_det + D + D * math.log(2*math.pi))
    conditional_lp = data_likelihoods.log_prob(data).detach()

    eig = ((conditional_lp).sum(0) / N) - marginal_lp
    
    out_dict = {'N': N}
    
    return eig, out_dict


def variational_marginal(
    self,
    design,
    guide,
    N,
    M=-1,
    guide_kwargs={},
    n_batch=1,
    n_epochs=100,
    optimizer=None,
    optimizer_kwargs={},
    scheduler=None,
    scheduler_kwargs={},
    return_guide=True,
    return_train_loss=True,
    return_test_loss=True,
    progress_bar=False,
    worker_id=None):
       
    M = M if not M == -1 else self.n_prior - N
    
    # Some checks to make sure all inputs are correct
    if n_batch > M: 
        raise ValueError(
        'Batch size cant be larger than M. Choose a smaller batch size or a larger M')

    data_samples = self.get_forward_samples(design, N+M)
    
    if data_samples == None:
        return torch.nan, None
    
    M_data_samples = data_samples[:M]
    N_data_samples = data_samples[M:]

    d_dicts = {n: self.design_dicts[n] for n in design}
    likelihood_kwargs = {'design': design, 'd_dicts': d_dicts}
    
    M_data_likelihoods = self.data_likelihood(M_data_samples, **likelihood_kwargs)
    N_data_likelihoods = self.data_likelihood(N_data_samples, **likelihood_kwargs)
    
    if N_data_likelihoods == None or M_data_likelihoods == None:
        raise ValueError('Likelihood not defined for this design. EIG can not be calculated with the variational marginal method')

    M_data_samples = M_data_likelihoods.sample()
    N_data_samples = N_data_likelihoods.sample()
    
    M_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(M_data_samples),
        batch_size=n_batch, shuffle=True)
    
    guide = guide(M_data_samples, **guide_kwargs)
    
    if optimizer is None:
        optimizer = torch.optim.Adam(guide.parameters(), **optimizer_kwargs)
    else:
        optimizer = optimizer(guide.parameters(), **optimizer_kwargs)
        
    if scheduler is not None:
            scheduler = scheduler(optimizer, **scheduler_kwargs)
    else:
        class DummyScheduler:
            def step(self):
                pass
        scheduler = DummyScheduler()
        
    train_loss_list = []
    test_loss_list  = []
    
    if progress_bar:
        pbar = tqdm(range(n_epochs), desc='Epoch 0/0, Loss: 0.000')
        
    for e in range(n_epochs):
        
        for batch_ndx, batch in enumerate(M_dataloader):
            batch = batch[0].detach()
            optimizer.zero_grad()
            loss = -guide.log_prob(batch).mean()
            loss.backward()
            optimizer.step()
            
            train_loss_list.append(loss.item())

        with torch.no_grad():
            l = -guide.log_prob(N_data_samples).mean().item()
        test_loss_list.append(l)        
        scheduler.step()

        if progress_bar:                          
            pbar.set_description(f"Epoch {e+1}/{n_epochs}, Loss: {l:.3f}")
            pbar.update(1)

    if progress_bar:
        pbar.close()
            
    marginal_lp = guide.log_prob(N_data_samples).detach()
    conditional_lp = N_data_likelihoods.log_prob(N_data_samples).detach()
    
    eig = ((conditional_lp - marginal_lp).sum(0) / N)
    
    out_dict = {
        'N': N, 'M': M,
        'n_epochs': n_epochs, 'n_batch': n_batch,
        'optimizer_kwargs': optimizer_kwargs,
        'scheduler_kwargs': scheduler_kwargs}
    
    if return_guide or return_train_loss or return_test_loss:
        if return_guide:
            out_dict['guide'] = guide
        if return_train_loss:
            out_dict['train_loss'] = train_loss_list
        if return_test_loss:
            out_dict['test_loss']  = test_loss_list
            
    return eig, out_dict        
        

def variational_posterior(
    self,
    design,
    guide,
    N,
    M=-1,
    guide_kwargs={},
    n_batch=1,
    n_epochs=100,
    optimizer=None,
    optimizer_kwargs={},
    scheduler=None,
    scheduler_kwargs={},
    interrogation_mapping=None,
    prior_entropy = None,
    return_guide=True,
    return_train_loss=True,
    return_test_loss=True,
    progress_bar=False,
    worker_id=None):
    
    M = M if not M == -1 else self.n_prior - N
    
    # Some checks to make sure all inputs are correct
    if n_batch > M: 
        raise ValueError(
        'Batch size cant be larger than M. Choose a smaller batch size or a larger M')
        
    data_samples = self.get_forward_samples(design, N+M)
    if data_samples == None:
        return torch.nan, None
                    
    model_samples = self.prior_samples[:N+M]
    
    if interrogation_mapping is not None:
        model_samples = interrogation_mapping(model_samples)
        prior_entropy = prior_entropy if prior_entropy is not None else 0
        # print('Entropy of interrogation mapping not implemented yet')
    else:
        prior_entropy = self.prior_entropy
    
    M_model_samples = model_samples[:M]
    N_model_samples = model_samples[M:]
    
    M_data_samples = data_samples[:M]
    N_data_samples = data_samples[M:]
    
    d_dicts = {n: self.design_dicts[n] for n in design}
    likelihood_kwargs = {'design': design, 'd_dicts': d_dicts}
    
    if self.data_likelihood is not None:
        M_data_likelihood = self.data_likelihood(M_data_samples, **likelihood_kwargs)
        N_data_likelihood = self.data_likelihood(N_data_samples, **likelihood_kwargs)
    
        M_data_samples = M_data_likelihood.sample()
        N_data_samples = N_data_likelihood.sample()
        
    M_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(M_model_samples, M_data_samples),
        batch_size=n_batch, shuffle=True)
    
    guide = guide(M_model_samples, M_data_samples, **guide_kwargs)
    
    if optimizer is None:
        optimizer = torch.optim.Adam(guide.parameters(), **optimizer_kwargs)
    else:
        optimizer = optimizer(guide.parameters(), **optimizer_kwargs)

    if scheduler is not None:
            scheduler = scheduler(optimizer, **scheduler_kwargs)
    else:
        class DummyScheduler:
            def step(self):
                pass
        scheduler = DummyScheduler()
        
    train_loss_list = []
    test_loss_list  = []
    
    if progress_bar:
        pbar = tqdm(range(n_epochs), desc='Epoch 0/0, Loss: 0.000')
    
    for e in range(n_epochs):
        
        for batch_ndx, batch in enumerate(M_dataloader):
            
            model_batch = batch[0].detach()
            data_batch  = batch[1].detach()

            optimizer.zero_grad()
            loss = -guide.log_prob(model_batch, data_batch).mean()
            loss.backward()
            optimizer.step()
            
            train_loss_list.append(loss.item())
        
        with torch.no_grad():
            l = -guide.log_prob(N_model_samples, N_data_samples).mean().item()
        test_loss_list.append(l)
        scheduler.step()

        if progress_bar:                        
            pbar.set_description(f"Epoch {e+1}/{n_epochs}, Loss: {l:.3f}")
            pbar.update(1)

    if progress_bar:
        pbar.close()
        
    marginal_lp = guide.log_prob(N_model_samples, N_data_samples).detach()

    eig = prior_entropy + (marginal_lp.sum(0) / N)

    out_dict = {'N': N, 'M': M,
                'n_epochs': n_epochs, 'n_batch': n_batch,
                'optimizer_kwargs': optimizer_kwargs,
                'scheduler_kwargs': scheduler_kwargs}

    if return_guide or return_train_loss or return_test_loss:
        if return_guide:
            out_dict['guide'] = guide
        if return_train_loss:
            out_dict['train_loss'] = train_loss_list
        if return_test_loss:
            out_dict['test_loss']  = test_loss_list
            
    return eig, out_dict        
        
    