import torch
import numpy as np
from tqdm.autonotebook import tqdm

def iterative_construction(
    self, design_point_names, design_budget,
    eig_method, eig_method_kwargs,
    num_workers=1, progress_bar=True,
    initial_design=[],
    allow_repeats=True,
    **kwargs):
    
    design_dicts = {n: self.design_dicts[n] for n in design_point_names}
    
    cost_list = [c['cost'] for n, c in design_dicts.items()]
    if len(set(cost_list)) > 1:
        raise ValueError('Costs of design points have to be equal! Otherwise the algorithm does not work!')
    
    budget = 0
     
    out_dict = {}
        
    with tqdm(total=design_budget, disable=(not progress_bar), position=0) as pbar:
        
        count = 1
        
        optimal_design = initial_design.copy()
                
        while budget < design_budget:
            
            if allow_repeats:
                temp_design_names = [optimal_design + [n] for n in design_point_names]
            else:
                temp_design_names = []
                for n in design_point_names:
                    if n not in optimal_design:
                        temp_design_names.append(optimal_design + [n])
                    else:
                        temp_design_names.append(optimal_design)
                        
            #TODO: add check if no data is returned e.g.: interstation designs with one receiver
            
            eig_list_kwargs = {}
            if 'progress_bar_eig' in eig_method_kwargs:
                eig_list_kwargs['progress_bar'] = eig_method_kwargs['progress_bar_eig']
            if 'random_seed' in eig_method_kwargs:
                eig_list_kwargs['random_seed'] = eig_method_kwargs['random_seed']
            if 'parallel_method' in eig_method_kwargs:
                eig_list_kwargs['parallel_method'] = eig_method_kwargs['parallel_method']               
            
            out_list = self.calculate_eig_list(temp_design_names, eig_method, eig_method_kwargs, num_workers, **eig_list_kwargs)
            eig_list, info_list = out_list
            eig_list = eig_list.detach().numpy()
            
            if np.all(np.isnan(eig_list)):
                raise ValueError('All EIG values are nan! Check if enough design points are predefined for an iteriv design to work!')
            
            new_design_point = design_point_names[np.nanargmax(eig_list)]
            budget += design_dicts[new_design_point]['cost']
            optimal_design.append(new_design_point)
            
            pbar.update(design_dicts[new_design_point]['cost'])
            
            out_dict[count] = {}
            out_dict[count]['eig'] = eig_list
            
            if all([guide != None for guide in info_list]):
                out_dict[count]['info'] = info_list

            count += 1
                    
    return optimal_design, out_dict
