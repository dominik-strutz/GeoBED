import h5py
import numpy as np


class dataloader():
    def __init__(self, filename):
        self.filename = filename
    def __enter__(self):
        #ttysetattr etc goes here before opening and returning the file object        
        if self.filename.endswith('.hdf5'):
            self.fd = h5py.File((self.filename), "r")
        elif self.filename.endswith('.npz'):
            self.npz = open(self.filename, 'rb')
            self.fd = np.load(self.npz)
        else:
            raise ValueError('Filetype not supported')
        
        return self.fd
    
    def __exit__(self, type, value, traceback):
        #Exception handling here
        if self.filename.endswith('.hdf5'):
            self.fd.close()
        elif self.filename.endswith('.npz'):
            self.npz.close()
