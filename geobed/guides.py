import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist

import zuko

# a lot more complicated than necessary since I had to reimplement zuko.flows.FlowModule since dill cant handle abc abstract classes
class GMM_guide(torch.nn.Module):
    def __init__(self, data_samples, **kwargs):
        torch.manual_seed(0)
        data_mean = torch.mean(data_samples, dim=0)
        data_std = torch.std(data_samples, dim=0)
        features = data_samples.shape[-1]
    
        super().__init__()
    
        self.base = zuko.flows.GMM(features=features, **kwargs)
        self.base.phi[1].data = (data_samples[:kwargs['components']]-data_mean)/data_std
                
        self.transforms = [zuko.flows.Unconditional(dist.AffineTransform, -data_mean/data_std, 1/data_std, buffer=True),]# buffer=True excludes the parameters from the optimization
    
    def forward(self):
                
        transform = zuko.transforms.ComposedTransform(*(t(None) for t in self.transforms))

        base = self.base(None)

        return zuko.flows.NormalizingFlow(transform, base)
        
    def log_prob(self, x):
                
        out = self.forward().log_prob(x)        
        return out
    
    def sample(self, n_samples):
        
        shape = torch.Size([n_samples])
        return self.forward().sample(shape) 


# a lot more complicated than necessary since I had to reimplement zuko.flows.FlowModule since dill cant handle abc abstract classes

class MDN_guide(torch.nn.Module):
    def __init__(self, model_samples, data_samples, **kwargs):
        super().__init__()
        torch.manual_seed(0)
        self.data_mean = torch.mean(data_samples, dim=0)
        self.data_std = torch.std(data_samples, dim=0)
        data_features = data_samples.shape[-1]
        
        model_mean = torch.mean(model_samples, dim=0)
        model_std = torch.std(model_samples, dim=0)
        model_features = model_samples.shape[-1]
    
        self.base = zuko.flows.GMM(features=model_features, context=data_features, **kwargs)
          
        min_index = self.base.sizes[0]
        max_index = self.base.sizes[0] + self.base.sizes[1]
        self.base.hyper[-1].bias.data[min_index:max_index]  = ((model_samples[:kwargs['components']]-model_mean)/model_std).flatten().detach()
        
        self.transforms = [zuko.flows.Unconditional(dist.AffineTransform, -model_mean/model_std, 1/model_std, buffer=True),]# buffer=True excludes the parameters from the optimization
    
    def forward(self, d=None):
        d = ((d-self.data_mean)/self.data_std).detach()
                    
        transform = zuko.transforms.ComposedTransform(*(t(d) for t in self.transforms))

        if d is None:
            base = self.base(d)
        else:
            base = self.base(d).expand(d.shape[:-1])

        return zuko.flows.NormalizingFlow(transform, base)
        
    def log_prob(self, m, d):
        
        out = self.forward(d).log_prob(m)
                
        return out
    
    def sample(self, d, n_samples):
        
        shape = torch.Size([n_samples])
        return self.forward(d).sample(shape).squeeze(0)
    

class FullyConnected(nn.Module):
    """
    Fully-connected neural network written as a child class of torch.nn.Module,
    used to compute the mutual information between two random variables.

    Attributes
    ----------
    self.fc_var1: torch.nn.Linear object
        Input layer for the first random variable.
    self.fc_var2: torch.nn.Linear object
        Input layer for the second random variable.
    self.layers: torch.nn.ModuleList object
        Object that contains all layers of the neural network.

    Methods
    -------
    forward:
        Forward pass through the fully-connected eural network.
    """

    def __init__(self, model_samples, data_samples, L=1, H=10):
        """
        Parameters
        ----------
        var1_dim: int
            Dimensions of the first random variable.
        var2_dim: int
            Dimensions of the second random variable.
        L: int
            Number of hidden layers of the neural network.
            (default is 1)
        H: int or np.ndarray
            Number of hidden units for each hidden layer. If 'H' is an int, all
            layers will have the same size. 'H' can also be an nd.ndarray,
            specifying the sizes of each hidden layer.
            (default is 10)
        """

        super(FullyConnected, self).__init__()

        torch.manual_seed(0)

        var1_dim = model_samples.shape[1]
        var2_dim = data_samples.shape[1]

        self.model_mean = torch.mean(model_samples, dim=0)
        self.model_std = torch.std(model_samples, dim=0)

        self.data_mean = torch.mean(data_samples, dim=0)
        self.data_std = torch.std(data_samples, dim=0)

        # check for the correct dimensions
        if isinstance(H, (list, np.ndarray)):
            assert len(H) == L, "Incorrect dimensions of hidden units."
            H = list(map(int, list(H)))
        else:
            H = [int(H) for _ in range(L)]

        self.fc_var1 = nn.Linear(var1_dim, H[0])
        self.fc_var2 = nn.Linear(var2_dim, H[0])

        # Define any further layers
        self.layers = nn.ModuleList()
        if L == 1:
            fc = nn.Linear(H[0], 1)
            self.layers.append(fc)
        elif L > 1:
            for idx in range(1, L):
                fc = nn.Linear(H[idx - 1], H[idx])
                self.layers.append(fc)
            fc = nn.Linear(H[-1], 1)
            self.layers.append(fc)
        else:
            raise ValueError('Incorrect value for number of layers.')

    def forward(self, model_samples, data_samples):
        """
        Forward pass through the neural network.

        Parameters
        ----------
        var1: torch.autograd.Variable
            First random variable.
        var2: torch.autograd.Variable
            Second random variable.
        """
        
        model_samples = (model_samples - self.model_mean) / self.model_std
        data_samples = (data_samples - self.data_mean) / self.data_std

        # Initial layer over random variables
        hidden = F.relu(self.fc_var1(model_samples) + self.fc_var2(data_samples))

        # All subsequent layers
        for idx in range(len(self.layers) - 1):
            hidden = F.relu(self.layers[idx](hidden))

        # Output layer
        output = self.layers[-1](hidden)

        return output
    
    def sample(self, var2):
        pass