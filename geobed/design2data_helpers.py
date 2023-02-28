import numpy as np
import torch
import h5py

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


def lookup_1to1_design(self, name_list, n_samples):
        
    name_list = list(dict.fromkeys(name_list)) # remove duplicates
        
    # list(set(name_list)) destroys the order of the list

    design_meta_list = [self.design_dicts[n] for n in name_list]
        
    n_prior = self.prior_samples.shape[0]
    data = torch.zeros((n_samples, len(name_list), 1))    
    
    filname_list = [design_meta['file'] for design_meta in design_meta_list]
    dataset_list = [design_meta['dataset'] for design_meta in design_meta_list]

    if len(list(set(filname_list))) == 1 and len(list(set(dataset_list))) == 1:
        index_list = [design_meta['index'] for design_meta in design_meta_list]
        with np.load(filname_list[0]) as df:
            data[:, :, :] = torch.from_numpy(df[dataset_list[0]][:n_samples, index_list, :])
    
    else:
        for i, design_meta in enumerate(design_meta_list):
            with np.load(design_meta['file']) as df:

                data[:, i, :] = torch.from_numpy(df[design_meta['dataset']][:n_samples, design_meta['index'], :])

    return data.flatten(start_dim=-2)



def lookup_interstation_design(self, name_list, n_samples):

    name_list = list(dict.fromkeys(name_list)) # remove duplicates 
    # list(set(name_list)) destroys the order of the list
    design_meta_list = [self.design_dicts[n] for n in name_list]

    n_design_points = len(self.design_dicts)

    if len(name_list) == 1:
        return None
    else:        
        indices = torch.tensor([d['index'] for d in design_meta_list])                
        indices = (torch.combinations(indices)).tolist()    
        indices = [list(sorted(i)) for i in indices]
                
        all_indices = zip(*torch.tril_indices(n_design_points, n_design_points, offset=-1).tolist())
        all_indices = [list(sorted(i)) for i in all_indices]
                
        indices = [i for i, ind in enumerate(all_indices) if ind in indices]
    
    n_prior = self.prior_samples.shape[0]
    data = torch.zeros((n_prior, len(indices), 1))    
    
    filename = design_meta_list[0]['file']
    dataset_name = design_meta_list[0]['dataset']
        
    with h5py.File(filename, "r") as df:
        data = torch.from_numpy(df[dataset_name][:n_samples, indices, :])
        
    return data.flatten(start_dim=-2)


def lookup_1to1_design_variable_length(self, name_list, n_samples):
        
    name_list = list(dict.fromkeys(name_list)) # remove duplicates 
    # list(set(name_list)) destroys the order of the list
    design_meta_list = [self.design_dicts[n] for n in name_list]

    data = []
    
    for i, design_meta in enumerate(design_meta_list):            
        
        with Dataloader(design_meta['file']) as df:
            i_data = np.stack(df[design_meta['dataset']][:n_samples, design_meta['index']])

            data.append(i_data)
                                      
    return torch.from_numpy(np.concatenate(data, axis=-1))