import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

import zuko

# a lot more complicated than necessary since I had to reimplement zuko.flows.FlowModule since dill cant handle abc abstract classes
class GMM_guide(torch.nn.Module):
    def __init__(
        self,
        data_samples,
        components,
        init_method=None,
        **kwargs):
        
        torch.manual_seed(0)
        data_mean = torch.mean(data_samples, dim=0)
        data_std = torch.std(data_samples, dim=0)
        features = data_samples.shape[-1]
    
        super().__init__()
    
        self.base = zuko.flows.GMM(features=features, components=components, **kwargs)
        
        if init_method == None:
            self.base.phi[1].data = (data_samples[:components]-data_mean)/data_std
        elif type(init_method) == dict:
            normalised_data_samples = (data_samples-data_mean)/data_std
            
            import warnings
            from sklearn.exceptions import ConvergenceWarning

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
            
                gm = GaussianMixture(
                    n_components=components,
                    random_state=0,
                    **init_method).fit(X=normalised_data_samples)
                
            self.base.phi[0].data = torch.tensor(gm.weights_).float()
            self.base.phi[1].data = torch.tensor(gm.means_).float()
            scale_tril = torch.linalg.cholesky(torch.tensor(gm.covariances_))
            # diagonal elements of the lower triangular matrix
            self.base.phi[2].data = torch.log(torch.diagonal(scale_tril, dim1=-2, dim2=-1)).float()
            # off diagonal elements of the lower triangular matrix
            self.base.phi[3].data = scale_tril[torch.tril(torch.ones_like(scale_tril, dtype=bool), diagonal=-1)].reshape(components, features*(features-1)//2).float()
                                 
        else:
            raise ValueError('Incorrect value for init_method.')
        
        self.transforms = [zuko.flows.Unconditional(dist.AffineTransform, -data_mean/data_std, 1/data_std, buffer=True),]# buffer=True excludes the parameters from the optimization
    
    def forward(self):
                
        transform = zuko.transforms.ComposedTransform(*(t(None) for t in self.transforms))

        base = self.base(None)

        return zuko.distributions.NormalizingFlow(transform, base)
        
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
          
        min_index = self.base.shapes[0][0]
        max_index = self.base.shapes[0][0] + self.base.shapes[1][0]*self.base.shapes[1][1]
        
        self.base.hyper[-1].bias.data[min_index:max_index]  = ((model_samples[:kwargs['components']]-model_mean)/model_std).flatten().detach()
        
        self.transforms = [zuko.flows.Unconditional(dist.AffineTransform, -model_mean/model_std, 1/model_std, buffer=True),]# buffer=True excludes the parameters from the optimization
     
    def forward(self, d=None):
        d = ((d-self.data_mean)/self.data_std).detach()
                    
        transform = zuko.transforms.ComposedTransform(*(t(d) for t in self.transforms))

        if d is None:
            base = self.base(d)
        else:
            base = self.base(d).expand(d.shape[:-1])

        return zuko.distributions.NormalizingFlow(transform, base)
        
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
    

class FullyConnected_FLO(nn.Module):
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

        super(FullyConnected_FLO, self).__init__()

        torch.manual_seed(0)

        self.var1_dim = model_samples.shape[1]
        self.var2_dim = data_samples.shape[1]
        self.input_dim = self.var1_dim + self.var2_dim

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

        self.fc_in = nn.Linear(self.input_dim, H[0])

        # Define any further layers
        self.layers = nn.ModuleList()
        if L == 1:
            fc = nn.Linear(H[0], 2)
            nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)
            self.layers.append(fc)
        elif L > 1:
            for idx in range(1, L):
                fc = nn.Linear(H[idx - 1], H[idx])
                nn.init.xavier_uniform_(fc.weight)
                nn.init.zeros_(fc.bias)
                self.layers.append(fc)
            fc = nn.Linear(H[-1], 2)
            nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)
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

        xy = torch.cat([model_samples, data_samples], dim=1)
        hidden = xy.view(xy.shape[0], self.input_dim)

        # Initial layer over random variables
        hidden = F.relu(self.fc_in(hidden))

        # All subsequent layers
        for idx in range(len(self.layers) - 1):
            hidden = F.relu(self.layers[idx](hidden))

        # Output layer
        output = self.layers[-1](hidden)

        return output
    
    def sample(self, var2):
        pass

class MLP_FLO(nn.Module):
    def __init__(self, x, y, H=[512,512], act_func=nn.ReLU()):
        super(MLP_FLO,self).__init__()
        input_dim = x.shape[-1] + y.shape[-1]
        output_dim = 2
        self.input_dim = input_dim
        self.hidden_dim = H
        self.output_dim = output_dim
        self.act_func = act_func
        
        
        self.x_mean = x.mean(dim=0)
        self.x_std = x.std(dim=0)
        
        self.y_mean = y.mean(dim=0)
        self.y_std = y.std(dim=0)
                
        layers = []
        for i in range(len(H)):
            if i==0:
                layer = nn.Linear(input_dim, H[i])
            else:
                layer = nn.Linear(H[i-1], H[i])
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.append(layer)
            #layers.append(nn.ReLU(True))
            layers.append(act_func)
        if len(H):                #if there is more than one hidden layer
            layer = nn.Linear(H[-1], output_dim)
        else:
            layer = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
        layers.append(layer)
        
        self._main = nn.Sequential(*layers)
        
    def forward(self, x, y):
        x = (x - self.x_mean)/self.x_std
        y = (y - self.y_mean)/self.y_std

        xy = torch.cat([x,y], dim=1)
        out = xy.view(xy.shape[0], self.input_dim)
        out = self._main(out)
        return out