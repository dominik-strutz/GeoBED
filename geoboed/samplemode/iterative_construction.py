
from faulthandler import disable
from tkinter import N
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessPool as Pool
# import multiprocessing as mp
# mp.set_start_method('forkserver')

from collections import defaultdict
from itertools import chain
from operator import methodcaller

def _find_optimal_design_iterative_construction(
    self, design_dim, boed_method, boed_method_kwargs,
    plot_loss=False, n_parallel=1,
    **optimization_method_kwargs):

    fixed_des_ind = []
    eig_array_dict = {}
    info_dict_dict = {}
    
    # save return dict before chaninging it in case it is needed to plot losses
    try:
        return_dict = boed_method_kwargs['return_dict'] 
    except KeyError:
        # only save return dict if it is given as an argument
        return_dict = False
    boed_method_kwargs['return_dict'] = True if (plot_loss or return_dict) else False
        
    for i_des in range(1, design_dim+1):
        
        if len(fixed_des_ind) > 0:
            
            design_indices = np.arange(0, self.n_designs, dtype=int)[:, None]

            design_list = [design_indices, ]
            for id_i in fixed_des_ind:
                design_list.insert(0, id_i*np.ones_like(design_indices))
                
            design_list = np.hstack(design_list)
            design_list = np.sort(design_list, axis=1) #list inidices must be in ascending order
        
        else:
            design_list = np.arange(0, self.n_designs, dtype=int)[:, None]
        
        if n_parallel > 1:
            
            def parallel_func(design_sublist):
                return self.get_eig(design_sublist, boed_method, boed_method_kwargs=boed_method_kwargs, disable_tqdm=True)
                        
            batch_size = len(design_list)//n_parallel+1
            # batch_size = 1
                                    
            with Pool(ncpus=n_parallel) as pool:
                out = pool.map(parallel_func, [design_list[x:x+batch_size] for x in range(0, len(design_list), batch_size)])
                
                eig_list = np.concatenate(list(zip(*out))[0])
                out_dict = (list(zip(*out))[1])
            
            # pool = Pool(ncpus=n_parallel)
            # out = pool.map(parallel_func, [design_list[x:x+batch_size] for x in range(0, len(design_list), batch_size)])
            
            # eig_list = np.concatenate(list(zip(*out))[0])
            # out_dict = (list(zip(*out))[1])
            
            # pool.close()
            # pool.join()
            # pool.restart() #pathos cant open new pools with same state as the old ones 
            
            #TODO: make check to see of right methods are used
            if return_dict or boed_method_kwargs['return_dict']:
                # initialise defaultdict of lists
                out_dict = defaultdict(list)
                
                # this fancy procesing is combing the output dicts from the parallel processes
                # only losses and var guide lists need to be combined 
                dict_items = map(methodcaller('items'), (list(zip(*out))[1]))
                for k, v in chain.from_iterable(dict_items):
                    if k == 'losses':
                        if type(out_dict[k]) == list:
                            out_dict[k] = v
                        else:
                            out_dict[k] = np.concatenate((out_dict[k], v), axis=1)
                    if k == 'var_guide' and return_dict:
                        out_dict[k].append(v)
                        
        # use serial processing that can make use of pytorch parallelization if n_parallel = 1 
        else:
            eig_list, out_dict = self.get_eig(design_list, boed_method, boed_method_kwargs=boed_method_kwargs)
                
        opteig_ind = eig_list.argmax()    
        fixed_des_ind.append(opteig_ind)
        
        eig_array_dict[f'{i_des}'] = eig_list
        if return_dict:
            info_dict_dict[f'{i_des}'] = out_dict

        print(f'Finished calculating EIG for design dimension {i_des}')

        if plot_loss and (boed_method == 'var_marg' or boed_method == 'var_post'):
            
            fig = plt.figure(figsize=(6, 3))
            ax_dict = fig.subplot_mosaic('a')
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(design_list)))
            for i_loss, loss in enumerate(out_dict['losses'].T):
                ax_dict['a'].plot(loss, color=colors[i_loss])
            
            ax_dict['a'].set_xlabel('Iteration')
            ax_dict['a'].set_ylabel('Loss')
            
            plt.show()
        
    if return_dict:
        return fixed_des_ind, eig_array_dict, info_dict_dict
    else:
        return fixed_des_ind, eig_array_dict