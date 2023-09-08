import math
import torch
import numpy as np

import tqdm

from torch.optim import Adam
from torch import Tensor
from .utils import DummyScheduler

def variational_posterior(
    self,
    design,
    guide,
    N,
    M=-1,
    guide_kwargs={},
    n_batch=1,
    n_epochs=100,
    optimizer=Adam,
    optimizer_kwargs={},
    scheduler=DummyScheduler,
    scheduler_kwargs={},
    return_guide=True,
    return_train_loss=True,
    return_test_loss=True,
    progress_bar=False,
    ):
    
    bound_kwargs = {}

    return _mi_lower_bound(
        self,
        design,
        guide,
        N,
        'variational_posterior',
        bound_kwargs,
        M,
        guide_kwargs,
        n_batch,
        n_epochs,
        optimizer,
        optimizer_kwargs,
        scheduler,
        scheduler_kwargs,
        return_guide,
        return_train_loss,
        return_test_loss,
        progress_bar,
        )
    
    
def minebed(
    self,
    design,
    guide,
    N,
    M=-1,
    guide_kwargs={},
    n_batch=1,
    n_epochs=100,
    optimizer=Adam,
    optimizer_kwargs={},
    scheduler=DummyScheduler,
    scheduler_kwargs={},
    return_guide=True,
    return_train_loss=True,
    return_test_loss=True,
    progress_bar=False,
    ):
    
    bound_kwargs = {}
    
    return _mi_lower_bound(
        self,
        design,
        guide,
        N,
        'minebed',
        bound_kwargs,
        M,
        guide_kwargs,
        n_batch,
        n_epochs,
        optimizer,
        optimizer_kwargs,
        scheduler,
        scheduler_kwargs,
        return_guide,
        return_train_loss,
        return_test_loss,
        progress_bar,
        )

def nce(
    self,
    design,
    guide,
    N,
    M=-1,
    K=None,
    guide_kwargs={},
    n_batch=1,
    n_epochs=100,
    optimizer=Adam,
    optimizer_kwargs={},
    scheduler=DummyScheduler,
    scheduler_kwargs={},
    return_guide=True,
    return_train_loss=True,
    return_test_loss=True,
    progress_bar=False,
    ):
    
    bound_kwargs={'K': K}
    
    return _mi_lower_bound(
        self,
        design,
        guide,
        N,
        'nce',
        bound_kwargs,
        M,
        guide_kwargs,
        n_batch,
        n_epochs,
        optimizer,
        optimizer_kwargs,
        scheduler,
        scheduler_kwargs,
        return_guide,
        return_train_loss,
        return_test_loss,
        progress_bar,
        )

def flo(
    self,
    design,
    guide,
    N,
    M=-1,
    guide_kwargs={},
    K=10,
    n_batch=1,
    n_epochs=100,
    optimizer=Adam,
    optimizer_kwargs={},
    scheduler=DummyScheduler,
    scheduler_kwargs={},
    return_guide=True,
    return_train_loss=True,
    return_test_loss=True,
    progress_bar=False,
    ):
    
    bound_kwargs={'K': K}
    
    return _mi_lower_bound(
        self,
        design,
        guide,
        N,
        'flo',
        bound_kwargs,
        M,
        guide_kwargs,
        n_batch,
        n_epochs,
        optimizer,
        optimizer_kwargs,
        scheduler,
        scheduler_kwargs,
        return_guide,
        return_train_loss,
        return_test_loss,
        progress_bar,
        )


def _mi_lower_bound(
    self,
    design,
    guide,
    N,
    bound='variational_posterior',
    bound_kwargs={},
    M=-1,
    guide_kwargs={},
    n_batch=1,
    n_epochs=100,
    optimizer=Adam,
    optimizer_kwargs={},
    scheduler=DummyScheduler,
    scheduler_kwargs={},
    return_guide=True,
    return_train_loss=True,
    return_test_loss=True,
    progress_bar=False,
    ):
    
    M = M if not M == -1 else self.n_prior - N
        
    # Some checks to make sure all inputs are correct
    if n_batch > M: 
        raise ValueError(
        'Batch size cant be larger than M. Choose a smaller batch size or a larger M')
        
    data_samples, model_samples, _ = self.get_forward_model_samples(
        design, n_samples_model=N+M, return_parameter_samples=True)
    
    #TODO maybe change this as this could cause confussion
    if self.target_forward_function is not None:
        model_samples = self.target_forward_function(model_samples)

    M_model_samples = model_samples[:M]
    N_model_samples = model_samples[M:]
    
    M_data_samples = data_samples[:M]
    N_data_samples = data_samples[M:]    

    M_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(model_samples[:M], data_samples[:M]),
        batch_size=n_batch, shuffle=True)
    
    if bound == 'variational_posterior':
        loss_function = _variational_posterior_loss
    elif bound == 'minebed':
        loss_function = _minebed_loss
    elif bound == 'nce':
        loss_function = _nce_loss
    elif bound == 'flo':
        loss_function = _FLO_loss
    else:
        raise NotImplementedError('Bound not implemented yet')
    
    guide = guide(M_model_samples, M_data_samples, **guide_kwargs)
    optimizer = optimizer(guide.parameters(), **optimizer_kwargs)
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
            
            loss = -1*loss_function(guide, model_batch, data_batch, **bound_kwargs) 
                        
            loss.backward()
            optimizer.step()
            
            train_loss_list.append(loss.item())
        
        with torch.no_grad():            
            l = -1*loss_function(guide, N_model_samples, N_data_samples, **bound_kwargs).item()
        test_loss_list.append(l)
        scheduler.step()

        if progress_bar:                        
            pbar.set_description(f"Epoch {e+1}/{n_epochs}, Loss: {l:.3f}")
            pbar.update(1)

    if progress_bar:
        pbar.close()
         
    eig = loss_function(guide, N_model_samples, N_data_samples, prior_entropy=self.m_prior_dist_entropy, **bound_kwargs).detach()

    out_dict = {'N': N, 'M': M,
                'n_epochs': n_epochs, 'n_batch': n_batch,
                'optimizer': optimizer.__class__.__name__,
                'optimizer_kwargs': optimizer_kwargs,
                'scheduler': scheduler.__class__.__name__,
                'scheduler_kwargs': scheduler_kwargs}

    if return_guide:
        out_dict['guide'] = guide
        out_dict['guide_kwargs'] = guide_kwargs
    if return_train_loss:
        out_dict['train_loss'] = train_loss_list
    if return_test_loss:
        out_dict['test_loss']  = test_loss_list
            
    return eig, out_dict
        
    
def _variational_posterior_loss(guide, model_batch, data_batch, prior_entropy=None):
    
    if prior_entropy is not None:
        return prior_entropy + guide.log_prob(model_batch, data_batch).mean(dim=0)
    else:
        return guide.log_prob(model_batch, data_batch).mean(dim=0)


def _minebed_loss(guide, model_batch, data_batch, prior_entropy=None):
    '''
    Taken from: https://github.com/stevenkleinegesse/GradBED/blob/main/gradbed/bounds/nwj.py
    '''
    
    # Shuffle y-data for the second expectation
    idxs = np.random.choice(
        range(len(data_batch)), size=len(data_batch), replace=False)
    # We need y_shuffle attached to the design d
    y_shuffle = data_batch[idxs]

    # Get predictions from network
    pred_joint = guide(model_batch, data_batch)
    pred_marginals = guide(model_batch, y_shuffle)

    # Compute the MINE-f (or NWJ) lower bound
    Z = torch.tensor(np.exp(1), dtype=torch.float)
    mi_ma = torch.mean(pred_joint) - torch.mean(
        torch.exp(pred_marginals) / Z + torch.log(Z) - 1)

    # we want to maximize the lower bound; PyTorch minimizes
    return mi_ma

def _nce_loss(guide, x_sample, y_sample, prior_entropy=None, mode='posterior', K=None):
    
    if K is None:
        if mode == 'likelihood':
            sum_dim = 0
        elif mode == 'posterior':
            sum_dim = 1
        else:
            raise NotImplemented("Choose either 'likelihood' or 'posterior'.")

        batch_size = x_sample.size(0)
        K = K if K is not None else batch_size
        
        # Tile all possible combinations of x and y
        x_stacked = torch.stack(
            [x_sample] * K, dim=0).reshape(batch_size * K, -1)
        y_stacked = torch.stack(
            [y_sample] * K, dim=1).reshape(batch_size * K, -1)

        # get model predictions for joint data
        pred = guide(x_stacked, y_stacked).reshape(K, batch_size).T

        # rows of pred correspond to values of x
        # columns of pred correspond to values of y

        # log batch_size
        logK = torch.log(torch.tensor(pred.size(0), dtype=torch.float))

        # compute MI
        kl = torch.diag(pred) - torch.logsumexp(pred, dim=sum_dim) + logK
        mi = torch.mean(kl)

        return mi

    else:
        
        y_list = [y_sample]
        for k in range(K-1):
            y0 = y_sample[torch.randperm(y_sample.size()[0])]
            y_list.append(y0)
        
        log_fxy = []
        
        K = len(y_list)
        
        for y in y_list:
            log_fxy.append(guide(x_sample, y))
        log_fxy0 = log_fxy[0]   
        log_fxy = torch.cat(log_fxy,1)
        out = log_fxy0 - torch.logsumexp(log_fxy, dim=1, keepdim=True) + torch.log(torch.Tensor([K]))
                
        mi = torch.mean(out)

        return mi


def _FLO_loss(guide, x_sample, y_sample, prior_entropy=None, K=10):
    '''taken from: https://github.com/qingguo666/FLO'''
    
    x = x_sample
    y = y_sample
       
         
    # if K == 1:
    #     K = None
    if K == 'full':
        K = y.size(0)
    if K == 'adaptive':
        if y.size(0) > 100:
            K = 10
        else:
            K = 1

         
    # gu = guide(x,y)
    # if isinstance(gu, tuple):
    #     hx,hy,u = gu
    #     similarity_matrix = hx @ hy.t()
    #     pos_mask = torch.eye(hx.size(0),dtype=torch.bool)
    #     g = similarity_matrix[pos_mask].view(hx.size(0),-1)
    #     g0 = similarity_matrix[~pos_mask].view(hx.size(0),-1)
    #     g0_logsumexp = torch.logsumexp(g0,1).view(-1,1)
    #     output = u + torch.exp(-u+g0_logsumexp-g)/(hx.size(0)-1) - 1

    # else:
    #     g, u = torch.chunk(guide(x,y),2,dim=1)
    #     y0 = y[torch.randperm(y.size()[0])]
    #     if K is not None:

    #         for k in range(K-1):

    #             if k==0:
    #                 g0,_ = torch.chunk(guide(x,y0),2,dim=1)
    #             else:
    #                 y0 = y[torch.randperm(y.size()[0])]
    #                 g00,_ = torch.chunk(guide(x,y0),2,dim=1)
    #                 g0 = torch.cat((g0,g00),1)

    #         g0_logsumexp = torch.logsumexp(g0,1).view(-1,1)
    #         output = u + torch.exp(-u+g0_logsumexp-g)/(K-1) - 1
    #     else:    

    #         g0, _ = torch.chunk(guide(x,y0),2,dim=1)
    #         output = u + torch.exp(-u+g0-g) - 1    
    
    # return -torch.mean(output)
    
    g, u = torch.chunk(guide(x,y),2,dim=1)
    mi = 0
    
    for k in range(K):
        y0 = y[torch.randperm(y.size()[0])]
        g0, _ = torch.chunk(guide(x,y0),2,dim=1)
        output = torch.exp(g0-g)
    
    mi = mi/K
        
    return -(u + torch.exp(-u)*output - 1).mean()
    
    # Works
    # mi = 0
    # for k in range(K):
    #     y0 = y[torch.randperm(y.size()[0])]
    #     g, u = torch.chunk(guide(x,y),2,dim=1)
    #     g0, _ = torch.chunk(guide(x,y0),2,dim=1)
    #     output = u + torch.exp(-u+g0-g) - 1
        
    #     mi += output.mean()
        
    # return -(mi/K)  
