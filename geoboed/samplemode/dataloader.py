import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Dataloader():
    def __init__(self, filename):
        self.filename = filename
    def __enter__(self):
        #ttysetattr etc goes here before opening and returning the file object        
        if self.filename.endswith('.hdf5'):
            import h5py
            self.fd = h5py.File((self.filename), "r")
        elif self.filename.endswith('.npz'):
            self.npz = open(self.filename, 'rb')
            self.fd = np.load(self.npz, allow_pickle=True)
        else:
            raise ValueError('Filetype not supported')
        
        return self.fd
    
    def __exit__(self, type, value, traceback):
        #Exception handling here
        if self.filename.endswith('.hdf5'):
            self.fd.close()
        elif self.filename.endswith('.npz'):
            self.npz.close()


class DataPrepocessor(Dataset):

    def __init__(self, data_samples, model_samples=None, preprocessing=True):
        if not torch.is_tensor(data_samples):
            self.data = torch.from_numpy(data_samples)
        else:
            self.data = data_samples
        
        if model_samples is not None:
            if not torch.is_tensor(model_samples):
                self.model = torch.from_numpy(model_samples)
            else:
                self.model = model_samples
        else:
            self.model = None
                                            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.model is not None:
            return self.data[idx], self.model[idx]
        else:
            return self.data[idx]