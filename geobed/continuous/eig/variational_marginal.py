import math
import torch
import numpy as np

import tqdm

from torch.optim import Adam
from torch import Tensor
from .utils import DummyScheduler

def variational_marginal(
    self,
    design,
    guide,
    N,
    M,
    guide_kwargs={},
    n_batch=1,
    n_epochs=1,
    optimizer=Adam,
    optimizer_kwargs={},
    scheduler=DummyScheduler,
    scheduler_kwargs={},
    return_guide=True,
    return_train_loss=True,
    return_test_loss=True,
    progress_bar=False,
    ):

    if (self.nuisance_dist is not None) \
        or self.target_forward_function \
            or self.implict_obs_noise_dist:
        raise ValueError("Variational marginal method method cannot be used with implicit likelihoods. Use the variational marginal likelihood method instead") 

    if n_batch > M: 
        raise ValueError(
        'Batch size cant be larger than M. Choose a smaller batch size or a larger M')
    
    N_data_samples, N_data_likelihoods = self.get_forward_model_samples(design, n_samples_model=N, return_distribution=True)
    M_data_samples, _ = self.get_forward_model_samples(design, n_samples_model=M, return_distribution=True)
    
    M_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(M_data_samples),
        batch_size=n_batch, shuffle=True)
    
    guide = guide(M_data_samples, **guide_kwargs)
    optimizer = optimizer(guide.parameters(), **optimizer_kwargs)
    scheduler = scheduler(optimizer, **scheduler_kwargs)
        
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
            scheduler.step()

        with torch.no_grad():
            l = -guide.log_prob(N_data_samples).mean().item()
        test_loss_list.append(l)        

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
        'optimizer': optimizer.__class__.__name__,  
        'optimizer_kwargs': optimizer_kwargs,
        'scheduler': scheduler.__class__.__name__,
        'scheduler_kwargs': scheduler_kwargs}
    
    if return_guide:
        out_dict['guide'] = guide
    if return_train_loss:
        out_dict['train_loss'] = train_loss_list
    if return_test_loss:
        out_dict['test_loss']  = test_loss_list
            
    return eig, out_dict        