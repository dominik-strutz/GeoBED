def check_batch_epoch(n_batch, n_epochs, N, M):
    if type(n_batch) != int:
        if type(n_batch) == float:
            n_batch = int(n_batch)
        elif callable(n_batch):
            n_batch = n_batch(**{'N':N, 'M':M})
        else:
            raise ValueError('n_batch must be int, float or callable')
    
    if type(n_epochs) != int:
        if type(n_epochs) == float:
            n_epochs = int(n_epochs)
        elif callable(n_epochs):
            n_epochs = n_epochs(**{'N':N, 'M':M, 'n_batch':n_batch})
        else:
            raise ValueError('n_epochs must be int, float or callable')
        
    return n_batch, n_epochs