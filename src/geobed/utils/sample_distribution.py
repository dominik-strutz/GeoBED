import torch
import logging

from torch import Tensor

class SampleDistribution():
    def __init__(self, samples: Tensor):
        self.samples = samples
        
        self.batch_shape = torch.Size([])
        
    def sample(self, sample_shape: torch.Size([]) = torch.Size([]) ):
        if isinstance(sample_shape, int):
            sample_shape = torch.Size([sample_shape,])
        shape = sample_shape + self.batch_shape
        n_samples = torch.prod(torch.tensor(shape))
        
        if n_samples > self.samples.size(0):
            raise ValueError('Number of samples requested is larger than the number of samples in the distribution.')
        
        indices = torch.randperm(self.samples.size(0))[:n_samples]
        out = self.samples[indices]
        out = out.reshape(list(sample_shape) + list(self.batch_shape) + list(self.samples.shape[-1:]))
        
        return out.squeeze(0)
    
    def expand(self, batch_shape):
        self.batch_shape = batch_shape
        return self  
    
    def _reset_sample_generator(self):
        self.iterator_counter = 0