import torch
import logging

from torch import Tensor

class SampleDistribution():
    def __init__(self, samples: Tensor):
        self.samples = samples
        self.iterator_counter = 0
        
        self.batch_shape = torch.Size([])
        
    def sample(self, sample_shape: torch.Size([]) = torch.Size([]) ):
        if type(sample_shape) == int:
            sample_shape = torch.Size([sample_shape,])
        shape = sample_shape + self.batch_shape        
        n_samples = torch.prod(torch.tensor(shape))
        
        out = self._sample_generator(N=n_samples)
        out = out.reshape(list(sample_shape) + list(self.batch_shape) + list(self.samples.shape[-1:]))
        return out.squeeze(0)
    
    def expand(self, batch_shape):
        self.batch_shape = batch_shape
        return self  
    
    def _sample_generator(self, N):
        # rewrite without generator to allow pickling
        if self.iterator_counter+N >= self.samples.shape[0]:
            raise ValueError('Not enough prior samples avaliable.')
        else:
            out = self.samples[self.iterator_counter:self.iterator_counter+N]
            self.iterator_counter += N
            return out
    
    def _reset_sample_generator(self):
        self.iterator_counter = 0
    
    def entropy(self):
        logging.info('Entropy of sample distribution is not defined. Setting it to zero.')
        return torch.tensor(0.)
        
    def log_prob(self, x):
        raise NotImplementedError('Log probability of sample distribution is not defined.')