import numpy as np
import torch

class Dataloader():
    def __init__(self, filename):
        self.filename = filename
    def __enter__(self):
        if self.filename.endswith('.hdf5') or self.filename.endswith('.h5'):
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


def lookup_1to1_fast(self, name_list, n_samples):
        
    index_list = [self.design_dicts[n]['index'] for n in name_list]
    design_meta_list = [self.design_dicts[n] for n in name_list]
    
    # h5 files cant deal with all forms of fancy indexing
    # see: https://stackoverflow.com/questions/38761878/indexing-a-large-3d-hdf5-dataset-for-subsetting-based-on-2d-condition
    name_list_h5 = list(dict.fromkeys(name_list)) # remove duplicates
    design_meta_list_h5 = [self.design_dicts[n] for n in name_list_h5]
        
    data = torch.zeros((n_samples, len(name_list), 1))    
    
    # check if all files are the same and all datasets are the same
    filname_list = [design_meta['file'] for design_meta in design_meta_list_h5]
    dataset_list = [design_meta['dataset'] for design_meta in design_meta_list_h5]

    if len(list(set(filname_list))) == 1 and len(list(set(dataset_list))) == 1:
        index_list_h5 = [design_meta['index'] for design_meta in design_meta_list_h5]
        index_list_h5_sorted = sorted(index_list_h5)
                
        expand_indices = [index_list_h5.index(i) for i in index_list]
                
        with Dataloader(filname_list[0]) as df:
                        
            data[:, :, :] = torch.from_numpy(df[dataset_list[0]][:n_samples, index_list_h5_sorted, :])[:, expand_indices, :]
    
    else:
        for i, design_meta in enumerate(design_meta_list):
            with Dataloader(design_meta['file']) as df:

                data[:, i, :] = torch.from_numpy(df[design_meta['dataset']][:n_samples, design_meta['index'], :])

    return data.flatten(start_dim=-2)

def constructor_1to1_fast(self, name_list, n_samples):
    
    design_dicts_list = [self.design_dicts[n] for n in name_list]
    data = torch.zeros((n_samples, len(name_list), 1))
                
    for i, d_meta in enumerate(design_dicts_list):
        data[:, i, :] = d_meta['forward_function'](d_meta, self.prior_samples[:n_samples])

    return data.flatten(start_dim=-2)

def lookup_interstation_design(self, name_list, n_samples):

    #TODO: probably cant handle station doubling byet

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
        
    with Dataloader(filename) as df:
        data = torch.from_numpy(df[dataset_name][:n_samples, indices, :])
        
    return data.flatten(start_dim=-2)


def lookup_1to1_design_flexible(self, name_list, n_samples):
        
    design_meta_list = [self.design_dicts[n] for n in name_list]

    data = []
    
    for i, design_meta in enumerate(design_meta_list):            
        
        with Dataloader(design_meta['file']) as df:
            i_data = np.stack(df[design_meta['dataset']][:n_samples, design_meta['index']])

            data.append(i_data)
                                      
    return torch.from_numpy(np.concatenate(data, axis=-1))