import math
import torch
import numpy as np

from tqdm.autonotebook import tqdm

from .utils import check_batch_epoch

def nmc(self, design, N, M=-1,
        reuse_M_samples=False,
        memory_efficient=False,
        worker_id=None):
    
    N = N if not N==-1 else self.n_prior
    M = M if not M==-1 else N
    
    if N < M: print("M should not be bigger than N! Asymptotically, M is idealy set to sqrt(N).")
    
    total_samples = N * (M+1) if not reuse_M_samples else N+M
    # if M_prime is not None:
    #     if independent_priors:
    #         total_samples += M_prime
    #     else:
    #         total_samples += (M_prime+1) * N
    
    data_samples = self.get_forward_samples(design, total_samples)

    if not memory_efficient:
    
        if data_samples == None:
            return torch.tensor(torch.nan), None

        d_dicts = {n: self.design_dicts[n] for n in design}
        likelihood_kwargs = {'design': design, 'd_dicts': d_dicts}
        
        N_likelihoods = self.data_likelihood(data_samples[:N], **likelihood_kwargs)
        N_samples = N_likelihoods.sample()
        conditional_lp = N_likelihoods.log_prob(N_samples)
        
        if reuse_M_samples:        
            NM_samples = data_samples[N:, None]        
            NM_samples = NM_samples.expand(M, N, data_samples.shape[-1])
        else:
            NM_samples = data_samples[N:].reshape(M, N, data_samples.shape[-1])
                    
        NM_likelihoods = self.data_likelihood(NM_samples, **likelihood_kwargs)
        marginal_lp = NM_likelihoods.log_prob(N_samples).logsumexp(0) - math.log(M)
        
    else:
        
        if reuse_M_samples:
            N_samples = data_samples[:N]
            M_samples = data_samples[N:]
            
            d_dicts = {n: self.design_dicts[n] for n in design}
            likelihood_kwargs = {'design': design, 'd_dicts': d_dicts}
            
            N_likelihoods = self.data_likelihood(N_samples, **likelihood_kwargs)
            N_samples = N_likelihoods.sample()
            conditional_lp = N_likelihoods.log_prob(N_samples)
            
            marginal_lp = torch.zeros(N)
            
            for i in range(N):
                marginal_lp[i] = self.data_likelihood(M_samples, **likelihood_kwargs).log_prob(N_samples[i]).logsumexp(0) - math.log(M)
            
        else:
            raise NotImplementedError('Memory efficient mode does not only reuse_M_samples')
        
    eig = (conditional_lp - marginal_lp).sum(0) / N
    
    out_dict = {'N': N, 'M': M, 'reuse_N_samples': reuse_M_samples}
    
    return eig, out_dict


def dn(self, design, N=None, worker_id=None):

    #TODO: Implement test for gaussian noise and design independent noise
    #TODO: Implement determinant of std errors for design dependent gaussian
    
    N = N if not N == None else self.n_prior
    
    data_samples = self.get_forward_samples(design, N)
    
    if data_samples == None:
        return torch.tensor(torch.nan), None

    d_dicts = {n: self.design_dicts[n] for n in design}
    likelihood_kwargs = {'design': design, 'd_dicts': d_dicts}
    
    data_likelihoods = self.data_likelihood(data_samples, **likelihood_kwargs)
    
    if data_likelihoods == None:
        raise ValueError('Likelihood not defined for this design. EIG an not be calculated with the DN method')
    
    data = data_likelihoods.sample()
    
    if data == None:
        return torch.tensor(torch.nan), None
    
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
    
    n_batch, n_epochs = check_batch_epoch(n_batch, n_epochs, N, M)
        
    # Some checks to make sure all inputs are correct
    if n_batch > M: 
        raise ValueError(
        'Batch size cant be larger than M. Choose a smaller batch size or a larger M')

    if self.data_likelihood is None:
        raise ValueError('Likelihood not defined for this design. EIG can not be calculated with the variational marginal method')

    data_samples = self.get_forward_samples(design, N+M)    
    
    if data_samples == None:
        return torch.tensor(torch.nan), None
    
    M_data_samples = data_samples[:M]
    N_data_samples = data_samples[M:]

    d_dicts = {n: self.design_dicts[n] for n in design}
    likelihood_kwargs = {'design': design, 'd_dicts': d_dicts}
    
    M_data_likelihoods = self.data_likelihood(M_data_samples, **likelihood_kwargs)
    N_data_likelihoods = self.data_likelihood(N_data_samples, **likelihood_kwargs)
    
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
        for key in scheduler_kwargs:
            if callable(scheduler_kwargs[key]): 
                scheduler_kwargs[key] = scheduler_kwargs[key](**{'N': N, 'M': M, 'n_batch': n_batch, 'n_epochs': n_epochs})
        scheduler = scheduler(optimizer, **scheduler_kwargs)
    else:
        class DummyScheduler:
            def step(self):
                pass
        scheduler = DummyScheduler()
        
    train_loss_list = []
    test_loss_list  = []
    
    # print(f'N: {N}, M: {M}, n_batch: {n_batch}, n_epochs: {n_epochs}', scheduler, scheduler_kwargs, guide, guide_kwargs)
    
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
    
    bound_kwargs = {}

    return mi_lower_bound(
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
        interrogation_mapping,
        prior_entropy,
        return_guide,
        return_train_loss,
        return_test_loss,
        progress_bar,
        worker_id)
    
    
def minebed(
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
    
    bound_kwargs = {}
    
    return mi_lower_bound(
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
        interrogation_mapping,
        prior_entropy,
        return_guide,
        return_train_loss,
        return_test_loss,
        progress_bar,
        worker_id)

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
    
    bound_kwargs={'K': K}
    
    return mi_lower_bound(
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
        interrogation_mapping,
        prior_entropy,
        return_guide,
        return_train_loss,
        return_test_loss,
        progress_bar,
        worker_id)

def FLO(
    self,
    design,
    guide,
    N,
    M=-1,
    guide_kwargs={},
    K=None,
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
    
    bound_kwargs={'K': K}
    
    return mi_lower_bound(
        self,
        design,
        guide,
        N,
        'FLO',
        bound_kwargs,
        M,
        guide_kwargs,
        n_batch,
        n_epochs,
        optimizer,
        optimizer_kwargs,
        scheduler,
        scheduler_kwargs,
        interrogation_mapping,
        prior_entropy,
        return_guide,
        return_train_loss,
        return_test_loss,
        progress_bar,
        worker_id)


def mi_lower_bound(
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
    
    n_batch, n_epochs = check_batch_epoch(**{'N': N, 'M': M, 'n_batch': n_batch, 'n_epochs': n_epochs})
    
    # Some checks to make sure all inputs are correct
    if n_batch > M: 
        raise ValueError(
        'Batch size cant be larger than M. Choose a smaller batch size or a larger M')
        
    data_samples = self.get_forward_samples(design, N+M)
    if data_samples == None:
        return torch.tensor(torch.nan), None
                    
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
    
    if bound == 'variational_posterior':
        loss_function = variational_posterior_loss
    elif bound == 'minebed':
        loss_function = minebed_loss
    elif bound == 'nce':
        loss_function = nce_loss
    elif bound == 'FLO':
        loss_function = FLO_loss
    else:
        raise NotImplementedError('Bound not implemented yet')
    
    guide = guide(M_model_samples, M_data_samples, **guide_kwargs)

    if optimizer is None:
        optimizer = torch.optim.Adam(guide.parameters(), **optimizer_kwargs)
    else:
        optimizer = optimizer(guide.parameters(), **optimizer_kwargs)

    if scheduler is not None:
        for key in scheduler_kwargs:
            if callable(scheduler_kwargs[key]): 
                scheduler_kwargs[key] = scheduler_kwargs[key](**{'N': N, 'M': M, 'n_batch': n_batch, 'n_epochs': n_epochs})
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
         
    eig = loss_function(guide, N_model_samples, N_data_samples, prior_entropy=prior_entropy, **bound_kwargs).detach()

    out_dict = {'N': N, 'M': M,
                'n_epochs': n_epochs, 'n_batch': n_batch,
                'optimizer_kwargs': optimizer_kwargs,
                'scheduler_kwargs': scheduler_kwargs}

    if return_guide:
        out_dict['guide'] = guide
    if return_train_loss:
        out_dict['train_loss'] = train_loss_list
    if return_test_loss:
        out_dict['test_loss']  = test_loss_list
            
    return eig, out_dict
        
    
def variational_posterior_loss(guide, model_batch, data_batch, prior_entropy=None):
    
    if prior_entropy is not None:
        return prior_entropy + guide.log_prob(model_batch, data_batch).mean(dim=0)
    else:
        return guide.log_prob(model_batch, data_batch).mean(dim=0)


def minebed_loss(guide, model_batch, data_batch, prior_entropy=None):
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

def nce_loss(guide, x_sample, y_sample, prior_entropy=None, mode='posterior', K=None):
    
    
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


def FLO_loss(guide, x_sample, y_sample, prior_entropy=None, K=10):
    '''taken from: https://github.com/qingguo666/FLO'''
    
    x = x_sample
    y = y_sample
    
    K = 5
     
    gu = guide(x,y)
    if isinstance(gu, tuple):
        hx,hy,u = gu
        similarity_matrix = hx @ hy.t()
        pos_mask = torch.eye(hx.size(0),dtype=torch.bool)
        g = similarity_matrix[pos_mask].view(hx.size(0),-1)
        g0 = similarity_matrix[~pos_mask].view(hx.size(0),-1)
        g0_logsumexp = torch.logsumexp(g0,1).view(-1,1)
        output = u + torch.exp(-u+g0_logsumexp-g)/(hx.size(0)-1) - 1

    else:      
        g, u = torch.chunk(guide(x,y),2,dim=1)
        if K is not None:

            for k in range(K-1):

                if k==0:
                    y0 = y0
                    g0,_ = torch.chunk(guide(x,y0),2,dim=1)
                else:
                    y0 = y[torch.randperm(y.size()[0])]
                    g00,_ = torch.chunk(guide(x,y0),2,dim=1)
                    g0 = torch.cat((g0,g00),1)

            g0_logsumexp = torch.logsumexp(g0,1).view(-1,1)
            output = u + torch.exp(-u+g0_logsumexp-g)/(K-1) - 1
        else:    

            g0, _ = torch.chunk(guide(x,y0),2,dim=1)
            output = u + torch.exp(-u+g0-g) - 1
        return output
    
    
    return -torch.mean(output)
    
    
    
    # print(x_sample.shape)    
    # print(y_sample.shape)
    # print(y0.shape)
    
    # K=1
    
    # g = guide.critic(x_sample, y_sample)
    # u  = guide.u_func(x_sample, y_sample)
    
    # y0 = y_sample[torch.randperm(y_sample.size()[0])]
    # g0 = guide.critic(x_sample, y0)
    
    # for k in range(K-1):
    #     y0 = y_sample[torch.randperm(y_sample.size()[0])]
    #     g0 = torch.cat((g0,guide.critic(x_sample, y0)),1)
        
    # g0_logsumexp = torch.logsumexp(g0,1).view(-1,1)
    # output = u + torch.exp(-u+g0_logsumexp-g)/(K-1) - 1
    
    # return -torch.mean(output)