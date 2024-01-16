import math
import torch
import numpy as np

import tqdm

from torch.optim import Adam
from torch import Tensor
from .utils import DummyScheduler

def variational_marginal_likelihood(
    self,
    design,
    marginal_guide,
    conditional_guide,
    N,
    M,
    marginal_guide_kwargs={},
    conditional_guide_kwargs={},
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

    if n_batch > M: 
        raise ValueError(
        'Batch size cant be larger than M. Choose a smaller batch size or a larger M')

    # model_samples = self.get_m_prior_samples(N+M)    
    # nuisance_samples = self.get_nuisance_prior_samples(1, model_samples)

    if self.nuisance_dist:
        data_samples, model_samples = self.get_data_likelihood_samples(
            design, n_model_samples=N+M, n_nuisance_samples=1)
        data_samples = data_samples.squeeze(0)
        model_samples = model_samples.squeeze(0)
    else:
        data_samples, model_samples = self.get_data_likelihood_samples(
            design, n_model_samples=N+M)
    # #TODO maybe change this as this could cause confussion
    # if self.target_forward_function is not None:
    #     model_samples = self.target_forward_function(model_samples)
    
    if data_samples == None:
        return torch.tensor(torch.nan), None

    M_model_samples = model_samples[:M]
    N_model_samples = model_samples[M:]

    M_data_samples = data_samples[:M]
    N_data_samples = data_samples[M:]
    
    M_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(M_model_samples, M_data_samples),
        batch_size=n_batch, shuffle=True)
    
    conditional_guide = conditional_guide(M_model_samples, M_data_samples, **conditional_guide_kwargs)
    marginal_guide = marginal_guide(M_data_samples, **marginal_guide_kwargs)

    optimizer = optimizer(list(conditional_guide.parameters()) + list(marginal_guide.parameters()), **optimizer_kwargs)
    scheduler = scheduler(optimizer, **scheduler_kwargs)

        
    train_loss_list = []
    test_loss_list  = []
        
    if progress_bar:
        pbar = tqdm(range(n_epochs), desc='Epoch 0/0, Loss: 0.000')
        
    for e in range(n_epochs):
        
        for batch_ndx, batch in enumerate(M_dataloader):
            model_batch = batch[0].detach()
            data_batch  = batch[1].detach()
            
            optimizer.zero_grad()
                        
            loss = -conditional_guide.log_prob(data_batch, model_batch).mean() - marginal_guide.log_prob(data_batch).mean()
            loss.backward()
            optimizer.step()
            
            train_loss_list.append(loss.item())
            scheduler.step()

        with torch.no_grad():
            l = -conditional_guide.log_prob(data_batch, model_batch).mean() - marginal_guide.log_prob(data_batch).mean()
        test_loss_list.append(l)        

        if progress_bar:                          
            pbar.set_description(f"Epoch {e+1}/{n_epochs}, Loss: {l:.3f}")
            pbar.update(1)

    if progress_bar:
        pbar.close()
            
    eig = ((conditional_guide.log_prob(N_data_samples, N_model_samples) - marginal_guide.log_prob(N_data_samples)).sum(0) / N).detach()
        
    out_dict = {
        'N': N, 'M': M,
        'n_epochs': n_epochs, 'n_batch': n_batch,
        'optimizer': optimizer.__class__.__name__,  
        'optimizer_kwargs': optimizer_kwargs,
        'scheduler': scheduler.__class__.__name__,
        'scheduler_kwargs': scheduler_kwargs}

    if return_guide:
        out_dict['marginal_guide'] = marginal_guide
        out_dict['marignal_guide_kwargs'] = marginal_guide_kwargs
        out_dict['conditional_guide'] = conditional_guide
        out_dict['conditional_guide_kwargs'] = conditional_guide_kwargs
    if return_train_loss:
        out_dict['train_loss'] = train_loss_list
    if return_test_loss:
        out_dict['test_loss']  = test_loss_list
            
    return eig, out_dict        