import numpy as np
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from pyprop8.utils import stf_trapezoidal, rtf2xyz, make_moment_tensor

from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from scipy.interpolate import RegularGridInterpolator, interp1d

def dc2mt(DC_samples):
    
    MT_sampels = torch.zeros((DC_samples.shape[0], 6), dtype=torch.double)
    
    for i, DC_i in enumerate(DC_samples):
        
        strike = DC_i[0].item()
        dip    = DC_i[1].item()
        rake   = DC_i[2].item()
        M0     = DC_i[3].item()
        Mxyz = make_moment_tensor(strike, dip, rake, M0)
        
        MT_sampels[i, :] = torch.tensor(Mxyz)[[ [0, 1, 2, 0, 0, 1],  [0, 1, 2, 1, 2, 2] ] ]

    return MT_sampels 

class MT_Lookup_Class(nn.Module):
    
    try:
        import pyprop8 as pp
    except ImportError:
        raise ImportError('pyprop8 is required for this module!')
    
    def __init__(self, layers, receivers, source_location, source_time, n_t, t_0, t_N, mt_type='MT', derivatives=False, verbose=False, output_type='dis'):
        """Wrapper function for pyprop8 to compute lookup table for MT and derivatives.

        Args:
            layers (list): List of layers to be passed to pyprop8 LayeredStructureModel 
            receivers (np.array): Array of shape (n_receivers, 2) containing the receiver coordinates.
            source_location (np.array): Array of shape (3,) containing the source coordinates.
            source_time (float): Source time in seconds. 
            n_t (int): Number of time steps. 
            t_0 (float): Start time in seconds.
            t_N (float): End time in seconds.
        """        
        super(MT_Lookup_Class, self).__init__()

        self.mt_type = mt_type
        self.derivatives = derivatives
        self.output_type = output_type
        
        try:
            receivers = receivers.numpy()
        except:
            pass
        
        model = pp.LayeredStructureModel(layers)

        lrec = pp.ListOfReceivers(xx = receivers[:, 0], yy = receivers[:, 1], depth=0)

        MT_components = np.array(
            ((1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
             (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
             (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
             (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
             (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
             (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),)
            )

        Mxyz_list = []
        for _MT in MT_components:
            Mxyz = np.array(
                    ((_MT[0], _MT[3], _MT[4]),
                     (_MT[3], _MT[1], _MT[5]),
                     (_MT[4], _MT[5], _MT[2]),),
                )
            Mxyz_list.append(Mxyz)

        MT_components = np.array(Mxyz_list)

        F = np.zeros([MT_components.shape[0], 3, 1])

        stf = stf_trapezoidal
        stf_arg_list = (3, 6)

        source =  pp.PointSource(
            source_location[0], source_location[1], source_location[2],
            MT_components, F, source_time)

        t, dt = np.linspace(t_0, t_N, n_t, retstep=True)

        if verbose:
            print('Computing lookup table...')
            show_progress = True
        else:
            show_progress = False
        
        if derivatives:
            derivs = pp.DerivativeSwitches(
                moment_tensor=True
            )

            print('Computing lookup table...')
            self.tt, self.lookup_table, self.derivative_table = pp.compute_seismograms(
                model, source, lrec, n_t, dt,
                xyz=True,
                derivatives=derivs,
                squeeze_outputs=False,
                show_progress=True,
                # number_of_processes=10, # not in current version yet
                source_time_function=lambda w:stf(w, *stf_arg_list),
                )
            self.lookup_table = torch.tensor(self.lookup_table)
            self.derivative_table = torch.tensor(self.derivative_table)
            
        else:
        
            self.tt, self.lookup_table = pp.compute_seismograms(
                model, source, lrec, n_t, dt,
                xyz=True,
                squeeze_outputs=False,
                show_progress=show_progress,
                # number_of_processes=10, # not in current version yet
                source_time_function=lambda w:stf(w, *stf_arg_list),
                )
        
            self.lookup_table = torch.tensor(self.lookup_table)
        
    def forward(self, MT):
        """_summary_

        Args:
            MT (torch.tensor): Tensor of shape (batch_size, 6) containing the moment tensor components.

        Returns:
            tuple: Returns a tuple containing the seismograms and the derivatives with respect to the moment tensor components.
        """
        
        if self.mt_type == 'DC':
            MT = dc2mt(MT)
        
        if self.derivatives:
            seis        = torch.tensordot(MT, self.lookup_table, dims=([-1], [0]))
            # seismograms dont need to be weighted, derivatives do aparently
            derivatives = torch.tensordot(MT, self.derivative_table, dims=([-1], [0]))/MT.shape[0]  
            return seis, derivatives
        
        else:
            seis = torch.tensordot(MT, self.lookup_table, dims=([-1], [0]))
            
            if self.output_type == 'vel':
                seis = torch.diff(seis, prepend=torch.zeros(*seis.shape[:-1], 1))
            elif self.output_type == 'acc':
                seis = torch.diff(seis, n=2, prepend=torch.zeros(seis.shape[:-1], 2))
             
            return seis
        
    
class Calculate_Seimogram_FullSource(Function):

    try:
        import pyprop8 as pp
    except ImportError:
        raise ImportError('pyprop8 is required for this module!')
    
    @staticmethod
    def forward(
        ctx,
        MT,
        receivers,
        source_location,
        source_time,
        layers,
        n_t,
        t_0,
        t_N,
        derivative_dict={},
        show_progress=False,):
        
        #TODO: Make source time function a parameter
        
        if 'layers' in derivative_dict:
            raise NotImplementedError('Derivatives with respect to layers not implemented yet.')
        if 'r' in derivative_dict:
            raise NotImplementedError('Derivatives with respect to receivers radius not implemented yet.')
        if 'phi' in derivative_dict:
            raise NotImplementedError('Derivatives with respect to receivers azimuth not implemented yet.')
        if 'force' in derivative_dict:
            raise NotImplementedError('Derivatives with respect to force not implemented yet.')
        
        model = pp.LayeredStructureModel(layers)

        rec_coords = receivers.detach().numpy()
        rec_obj = pp.ListOfReceivers(xx = rec_coords[:, 0], yy = rec_coords[:, 1], depth=0)
        
        MT = MT.detach().numpy()
        Mxyz = np.stack(
            ((MT[..., 0], MT[..., 3], MT[..., 4]),
             (MT[..., 3], MT[..., 1], MT[..., 5]),
             (MT[..., 4], MT[..., 5], MT[..., 2]),), axis=0
        )
        
        Mxyz = np.moveaxis(Mxyz, -1, 0) # moves batch dimension to the begining, translation is of no effect if there is no batch dimension
                
        F = np.zeros([*Mxyz.shape[0:-1], 1])
        
        src_location = source_location.detach().numpy()
        src_time = source_time
        source_obj =  pp.PointSource(src_location[0], src_location[1], src_location[2], Mxyz, F, src_time)
        
        stf = stf_trapezoidal
        stf_arg_list = (3, 6)
        
        t, dt = np.linspace(t_0, t_N, n_t, retstep=True)
        
        _derivatives = {}
        
        if derivative_dict.get('moment_tensor', False):
            _derivatives['moment_tensor'] = True
        
        if derivative_dict.get('receiver_coordinates', False):
            _derivatives['r'] = True
            _derivatives['phi'] = True
        
        if derivative_dict.get('source_location', False):
            _derivatives['x'] = True
            _derivatives['y'] = True
            _derivatives['z'] = True

        if derivative_dict.get('source_time', False):
            _derivatives['source_time'] = True
                        
        _derivs = pp.DerivativeSwitches(
            **_derivatives
        )
                
        _, seis, derivatives = pp.compute_seismograms(
            model, source_obj, rec_obj, n_t, dt,
            xyz=True,
            derivatives=_derivs,
            squeeze_outputs=True,
            show_progress=show_progress,
            # number_of_processes=10, # not in current version yet
            source_time_function=lambda w:stf(w, *stf_arg_list),
            )
        
        # ctx is a context object that can be used to stash information
        # for backward computation
        derivatives = torch.tensor(derivatives).movedim(-3, -1)
        seis        = torch.tensor(seis       , requires_grad=True)
        
        derivative_bools = torch.zeros(4)
        
        if 'moment_tensor' in derivative_dict:
            derivative_bools[0] = 1
        if 'receiver_coordinates' in derivative_dict:
            derivative_bools[1] = 1
        if 'source_location' in derivative_dict:
            derivative_bools[2] = 1
        if 'time' in derivative_dict:
            derivative_bools[3] = 1
        
        ctx.save_for_backward(derivatives, receivers.detach(), derivative_bools.detach())
        ctx.mark_non_differentiable(derivatives)
                        
        return seis, derivatives
    
    @staticmethod
    @once_differentiable
    def backward(ctx, seis_grad_output, _derivatives_grad_output):
              
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None. 
              
        derivatives, rec_coords, derivative_bools = ctx.saved_tensors

        output = torch.tensordot(seis_grad_output.unsqueeze(-1), derivatives, dims=[(-4, -3, -2,), (-4, -3, -2,)])

        deriv_index = 0

        if derivative_bools[0] == 1:
            dMT = output[..., :6]
            deriv_index += 6
        else:
            dMT = None
            
        if derivative_bools[1] == 1:

            dRec_pol = output[..., deriv_index:deriv_index+2]
                                    
            # transform derivatives from cylindrical to cartesian
            r_vectors = torch.vstack((rec_coords[..., 0], rec_coords[..., 1], torch.zeros_like(rec_coords[..., 0]))).T        
            r_norms   = torch.linalg.norm(r_vectors, axis=-1)
            phi_vectors = torch.arctan2(rec_coords[..., 1], rec_coords[..., 0])
            
            d_recx = torch.cos(phi_vectors) * dRec_pol[..., 0] - torch.sin(phi_vectors) * dRec_pol[..., 1] / r_norms
            d_recy = torch.sin(phi_vectors) * dRec_pol[..., 0] + torch.cos(phi_vectors) * dRec_pol[..., 1] / r_norms
        
            dRec_cart = torch.stack((d_recx, d_recy), dim=-1)
            
            deriv_index += 2
        else:
            dRec_cart = None
        
        if derivative_bools[2] == 1:
            dLoc    = output[..., deriv_index:deriv_index+3]
            dLoc[..., 2] = -dLoc[..., 2] # depth is negative in pyprop8
            deriv_index += 3
                        
        else:
            dLoc = None
            
        if derivative_bools[3] == 1:            
            dTime = output[..., [deriv_index,]]
            deriv_index += 1
            
        else:
            dTime = None 
                           
        return dMT.detach(), dRec_cart.detach(), dLoc.detach(), dTime.detach(), None, None, None, None, None


class MTxyt_Lookup_Class(nn.Module):
    
    try:
        import pyprop8 as pp
    except ImportError:
        raise ImportError('pyprop8 is required for this module!')
    
    def __init__(self, layers, source_depth, grid_x, grid_y, n_t, t_0, t_N, max_shift, mt_type='MT', derivatives=False, verbose=False, number_of_processes=1):
       
        super(MTxyt_Lookup_Class, self).__init__()

        self.mt_type = mt_type
        
        self.n_t = n_t
        self.t_0 = t_0
        self.t_N = t_N
        self.max_shift = max_shift
        
        self.t_grid_out = np.linspace(t_0, t_N, n_t)
        self.n_t_internal = int(n_t*((t_N-t_0)+2*max_shift)/(t_N-t_0))
                
        self.t_grid_internal, self.dt_internal = np.linspace(t_0-max_shift, t_N+max_shift, self.n_t_internal, retstep=True)
                               
        try:
            receivers = receivers.numpy()
        except:
            pass
        
        if number_of_processes>1 and verbose:
            print('Progress bar for precomputation not supported with number_of_processes>1')
        
        model = pp.LayeredStructureModel(layers)
            
        grid_xx, grid_yy = np.meshgrid(grid_x, grid_y, indexing='ij')
        
        lrec = pp.ListOfReceivers(xx = grid_xx.flatten(), yy = grid_yy.flatten(), depth=0.0)

        MT_components = np.array(
            ((1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
             (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
             (0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
             (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
             (0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
             (0.0, 0.0, 0.0, 0.0, 0.0, 1.0),)
            )

        Mxyz_list = []
        for _MT in MT_components:
            Mxyz = np.array(
                    ((_MT[0], _MT[3], _MT[4]),
                     (_MT[3], _MT[1], _MT[5]),
                     (_MT[4], _MT[5], _MT[2]),),
                )
            Mxyz_list.append(Mxyz)

        MT_components = np.array(Mxyz_list)

        F = np.zeros([MT_components.shape[0], 3, 1])

        stf = stf_trapezoidal
        stf_arg_list = (3, 6)

        source =  pp.PointSource(
            0.0, 0.0, source_depth,
            MT_components, F, max_shift)

        if verbose:
            print('Computing lookup table...')
            show_progress = True
        else:
            show_progress = False
        
        if derivatives:
            
            raise NotImplementedError('Derivatives not implemented yet.')

            derivs = pp.DerivativeSwitches(
                moment_tensor=True
            )

            print('Computing lookup table...')
            self.tt, self.lookup_table, self.derivative_table = pp.compute_seismograms(
                model, source, lrec, self.n_t_internal, self.dt_internal,
                xyz=True,
                derivatives=derivs,
                squeeze_outputs=False,
                show_progress=True,
                # number_of_processes=10, # not in current version yet
                source_time_function=lambda w:stf(w, *stf_arg_list),
                )
            self.lookup_table = torch.tensor(self.lookup_table)
            self.derivative_table = torch.tensor(self.derivative_table)
            
        else:
                    
            self.tt, self.lookup_table = pp.compute_seismograms(
                model, source, lrec, self.n_t_internal, self.dt_internal,
                xyz=True,
                squeeze_outputs=False,
                show_progress=show_progress,
                number_of_processes=number_of_processes, # not in current version yet
                source_time_function=lambda w:stf(w, *stf_arg_list),
                )
                            
            self.lookup_table = torch.tensor(self.lookup_table).movedim(0, -1)
            
            self.interpolator = RegularGridInterpolator(
                (grid_x, grid_y),
                (self.lookup_table.flatten(-3, -2).reshape(grid_x.shape[0], grid_y.shape[0], self.n_t_internal*3, 6)).numpy(),
                method='linear')

    def forward(self, model_samples, receivers):
        """_summary_

        Args:
            MT (torch.tensor): Tensor of shape (batch_size, 6) containing the moment tensor components.

        Returns:
            tuple: Returns a tuple containing the seismograms and the derivatives with respect to the moment tensor components.
        """
        
        if not torch.is_tensor(model_samples):
            model_samples = torch.tensor(model_samples)
        if not torch.is_tensor(receivers):
            receivers = torch.tensor(receivers)
        
        model_samples_MT = torch.zeros((model_samples.shape[0], 9))

        if self.mt_type == 'DC':
            model_samples_MT[:, 3:] = dc2mt(model_samples[:, 3:])
        else:
            model_samples_MT = model_samples
        
        n_models = model_samples.shape[0]
        n_receivers = receivers.shape[0]
        
        horizontal_vector = model_samples_MT[:, :2].repeat_interleave(n_receivers, dim=0) - receivers.repeat(n_models, 1)
                
        interpolated_lookup = torch.tensor(self.interpolator(horizontal_vector.numpy())).reshape(n_models, n_receivers, 3, self.n_t_internal, 6)          
        
        seis_preshift = torch.einsum('im,ijklm->ijkl', model_samples_MT[:, 3:], interpolated_lookup)
                
        seis = torch.zeros((n_models, n_receivers, 3, self.n_t))
        for i_model, t_i in enumerate(model_samples_MT[:, 2]):
            seis[i_model] = torch.tensor(
                interp1d(self.t_grid_internal, seis_preshift[i_model].flatten(1, -2).numpy(), kind='cubic')(self.t_grid_out-t_i.numpy()))

        return seis
