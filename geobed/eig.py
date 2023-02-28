import math
import torch

import dill
dill.settings['recurse'] = True 

from tqdm.autonotebook import tqdm

def dn(self, design, N=None, worker_id=None):
                
    #TODO: Implement determinant of std errors for design dependent gaussian
    #TODO: Implement test for gaussian noise and design independent noise
    
    data_samples = self.get_forward_samples(design, N)
    
    if data_samples == None:
        return torch.nan, None

    d_dicts = {n: self.design_dicts[n] for n in design}
    likelihood_kwargs = {'design': design, 'd_dicts': d_dicts}
    
    data_likelihood = self.data_likelihood(data_samples, **likelihood_kwargs)
    
    if data_likelihood == None:
        raise ValueError('Likelihood not defined for this design. EIG an not be calculated with the DN method')
    
    data = data_likelihood.sample()
    
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
    conditional_lp = data_likelihood.log_prob(data).detach()

    eig = ((conditional_lp).sum(0) / N) - marginal_lp
    
    return eig, None


def variational_marginal(
    self,
    design,
    guide,
    N,
    M=-1,
    guide_kwargs={},
    n_batch=1,
    n_epochs=100,
    # optimizer=torch.optim.Adam,
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
    
    M_data_likelihood = self.data_likelihood(M_data_samples, **likelihood_kwargs)
    N_data_likelihood = self.data_likelihood(N_data_samples, **likelihood_kwargs)
    
    if N_data_likelihood == None or M_data_likelihood == None:
        raise ValueError('Likelihood not defined for this design. EIG an not be calculated with the variational marginal method')

    M_data_samples = M_data_likelihood.sample()
    N_data_samples = N_data_likelihood.sample()
    
    M_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(M_data_samples),
        batch_size=n_batch, shuffle=True)
    
    guide = guide(N_data_samples, **guide_kwargs)
    
    # optimizer = optimizer(guide.parameters(), **optimizer_kwargs)
    optimizer = torch.optim.Adam(guide.parameters(), **optimizer_kwargs)
        
    if scheduler is not None:
            scheduler = scheduler(optimizer, **scheduler_kwargs)
    else:
        class DummyScheduler:
            def step(self):
                pass
        scheduler = DummyScheduler()
        
    train_loss_list = []
    test_loss_list  = [] if return_test_loss else None
    
    if progress_bar:
        pbar = tqdm(range(n_epochs), desc='Epoch 0/0, Loss: 0.000')
        
    for e in range(n_epochs):
        
        for batch_ndx, batch in enumerate(M_dataloader):
            batch = batch[0]
            optimizer.zero_grad()
            loss = -guide.log_prob(batch).mean()
            loss.backward()
            optimizer.step()
            
            train_loss_list.append(loss.item())
        
        scheduler.step()
        
        if return_test_loss:
                        
            test_loss_list.append(-guide.log_prob(N_data_samples).mean().item())
            
        if progress_bar:
            
            l = train_loss_list[-1] if return_test_loss else test_loss_list[-1]
                 
            pbar.set_description(f"Epoch {e+1}/{n_epochs}, Loss: {l:.3f}")
            pbar.update(1)

    if progress_bar:
        pbar.close()
            
    marginal_lp = guide.log_prob(N_data_samples).detach()
    conditional_lp = N_data_likelihood.log_prob(N_data_samples).detach()
    
    eig = ((conditional_lp - marginal_lp).sum(0) / N)
    
    if return_guide or return_train_loss or return_test_loss:
        out_dict = {}
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
    # optimizer=torch.optim.Adam,
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
    
    if self.prior_dist == None:
        prior_entropy = torch.tensor(0.)
    else:
        try:
            prior_entropy = self.prior_dist.entropy()
        except:
            
            prior_entropy = torch.tensor(0.)
    
    model_samples = self.prior_samples[:N+M]
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
        torch.utils.data.TensorDataset(model_samples[:M], M_data_samples),
        batch_size=n_batch, shuffle=True)
    
    guide = guide(M_model_samples, M_data_samples, **guide_kwargs)
    
    # optimizer = optimizer(guide.parameters(), **optimizer_kwargs)
    optimizer = torch.optim.Adam(guide.parameters(), **optimizer_kwargs)
        
    if scheduler is not None:
            scheduler = scheduler(optimizer, **scheduler_kwargs)
    else:
        class DummyScheduler:
            def step(self):
                pass
        scheduler = DummyScheduler()
        
    train_loss_list = []
    test_loss_list  = [] if return_test_loss else None
    
    if progress_bar:
        pbar = tqdm(range(n_epochs), desc='Epoch 0/0, Loss: 0.000')
    
    for e in range(n_epochs):
        
        for batch_ndx, batch in enumerate(M_dataloader):
            
            model_batch = batch[0].clone().detach()
            data_batch  = batch[1].clone().detach()

            optimizer.zero_grad()
            loss = -guide.log_prob(model_batch, data_batch).mean()
            loss.backward()
            optimizer.step()
            
            train_loss_list.append(loss.item())
        
        scheduler.step()
        
        if return_test_loss:
                        
            test_loss_list.append(-guide.log_prob(N_model_samples, N_data_samples).mean().item())
            
        if progress_bar:
            
            l = train_loss_list[-1] if return_test_loss else test_loss_list[-1]
                                    
            pbar.set_description(f"Epoch {e+1}/{n_epochs}, Loss: {l:.3f}")
            pbar.update(1)

    if progress_bar:
        pbar.close()
        
    marginal_lp = guide.log_prob(N_model_samples, N_data_samples).detach()

    eig = prior_entropy + (marginal_lp.sum(0) / N)

    if return_guide or return_train_loss or return_test_loss:
        out_dict = {}
        if return_guide:
            out_dict['guide'] = guide
        if return_train_loss:
            out_dict['train_loss'] = train_loss_list
        if return_test_loss:
            out_dict['test_loss']  = test_loss_list
            
    return eig, out_dict        
        
    