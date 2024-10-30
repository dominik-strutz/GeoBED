import math
import torch
import numpy as np

import tqdm

from torch.optim import Adam
from torch import Tensor
from ..utils.misc import _DummyScheduler

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
    scheduler=_DummyScheduler,
    scheduler_kwargs={},
    return_guide=True,
    return_train_loss=True,
    return_test_loss=True,
    progress_bar=False,
    random_seed=None,
    ):

    if random_seed is not None:
        torch.manual_seed(random_seed)

    if self.nuisance_dist is not None:
        raise NotImplementedError("Variational marginal method not implemented yet for nuisance parameters")
    if self.implict_data_likelihood_func:
        raise ValueError("Variational marginal method cannot be used with implicit observation noise distribution")

    if n_batch > M: 
        raise ValueError(
        'Batch size cant be larger than M. Choose a smaller batch size or a larger M')
    
    N_data_likelihoods, _ = self.get_data_likelihood(
            design, n_model_samples=N)
    N_data_samples = N_data_likelihoods.sample()
    
    M_data_samples, _ = self.get_data_likelihood_samples(
            design, n_model_samples=M
    )
    
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
    
    eig = (conditional_lp - marginal_lp).detach()
    eig = eig.nansum(0) / torch.isfinite(eig).sum(0)

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