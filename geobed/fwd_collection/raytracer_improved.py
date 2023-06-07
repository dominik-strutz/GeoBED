import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LinearSegmentedColormap

try:
    import pykonal
    pykonal_installed = True
except ImportError:
    pykonal_installed = False
    
try:
    import ttcrpy
    from ttcrpy.rgrid import Grid2d
    ttcrpy_installed = True
except ImportError:
    ttcrpy_installed = False

class TTHelper_2D():
    def __init__(
        self,
        x,
        z,
        velocity_model,
        cell_velocity=True,
        coord_sys="cartesian"):
                
        self.coord_sys = coord_sys
        self.x = x
        self.z = z
        self.velocity_model = velocity_model
        self.cell_velocity = cell_velocity
        
        # helper variables for pykonal
        y = np.array([0])
        dx = x[1] - x[0]
        dy = 1.0
        dz = z[1] - z[0]
        
        self.x, self.y, self.z    = x, y, z
        self.min_coords           = min(x), min(y), min(z)
        self.node_intervals       = dx, dy, dz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.npts                 = len(x), len(y), len(z)
    
    def _forward_pykonal(
        self,
        sources, receivers,
        return_rays=False,
        return_tt_field=False,
        **kwargs):
        
        # add y coordinate
        receivers = np.vstack( [receivers[:, 0], np.zeros_like(receivers[:, 0]), receivers[:, 1]] ).T
        sources   = np.vstack( [sources[:, 0], np.zeros_like(sources[:, 0]), sources[:, 1]] ).T

        tt_list = np.zeros((len(sources), len(receivers)))
        ray_list_sources = []
        tt_field_list = []

        for i, src in enumerate(sources):
            
            solver = pykonal.solver.PointSourceSolver(coord_sys=self.coord_sys)
            solver.velocity.min_coords     = self.min_coords
            solver.velocity.node_intervals = self.node_intervals
            
            solver.velocity.npts           = self.npts
            solver.velocity.values         = self.velocity_model[:, None, :]
            solver.src_loc                 = np.array( (src[0], src[1], src[2]) ).astype('double')
            solver.solve()

            tt_list[i] = solver.traveltime.resample(receivers.astype('float'))

            if return_rays:
                
                ray_list_receivers = []
                
                for rec in receivers:
                    ray_coords = solver.traveltime.trace_ray(rec)                    
                    ray_list_receivers.append(ray_coords[:, [0, 2]])
                    
                ray_list_sources.append(ray_list_receivers)

            if return_tt_field:
                tt_field_list.append(solver.traveltime)
            
        out = [tt_list]
        if return_rays: out.append(ray_list_sources)
        if return_tt_field: out.append(tt_field_list)
            
        return out
    
    def _forward_ttcrpy(
        self,
        sources, receivers,
        return_rays=False,
        return_tt_field=False,
        **kwargs):
        
        cell_slowness = True if self.cell_velocity else False
        
        tt_list = np.zeros((len(sources), len(receivers)))
        ray_list_sources = []
        tt_field_list = []
        
        for i, src in enumerate(sources):
            
            src = src[None, :]
            
            ttcrpy_grid = Grid2d(self.x.astype('double'), self.z.astype('double'), cell_slowness=cell_slowness, **kwargs)        
            ttcrpy_grid.set_slowness(1/self.velocity_model.astype('double'))
        
            # ttrcpy works with source and receiver pairs so repeat sources and tile receivers to get all pairs
            N_src = src.shape[0]
            N_rcv = receivers.shape[0]
            src = src.repeat(N_rcv, 0)
            rcv = np.tile(receivers, (N_src, 1))
                            
            out = ttcrpy_grid.raytrace(src.astype('double'), rcv.astype('double'), return_rays=return_rays)
            
            if return_rays:
                tt, rays = out
                ray_list_sources.append(rays)
            else:
                tt = out
            tt_list[i] = tt
            
            if return_tt_field:
                            
                tt_field = pykonal.fields.ScalarField3D()
                
                tt_field.min_coords     = self.min_coords
                tt_field.node_intervals = self.node_intervals
                tt_field.npts           = self.npts
                tt_field.values         = ttcrpy_grid.get_grid_traveltimes()[:, None, :]
                tt_field_list.append(tt_field)
            
            
        out = [tt_list]
        if return_rays: out.append(ray_list_sources)
        if return_tt_field: out.append(tt_field_list)
            
        return out
        
    def calculate_tt(
        self, sources, receivers,
        return_rays=False,
        return_tt_field=False,
        solver='pykonal',
        **kwargs):
        
        sources = np.array(sources) if not type(sources) == torch.Tensor else sources.numpy()
        if len(sources.shape) == 1:
            sources = sources[None, :]
            
        receivers = np.array(receivers) if not type(receivers) == torch.Tensor else receivers.numpy()
        if len(receivers.shape) == 1:
            receivers = receivers[None, :]
        
        if solver == 'pykonal':
            if not pykonal_installed:
                raise ImportError('Pykonal is not installed.')
            
            if self.cell_velocity:
                raise ValueError('Pykonal only supports node velocities.')

            return self._forward_pykonal(
                sources, receivers, return_rays=return_rays, return_tt_field=return_tt_field, **kwargs)
        
        
        if solver == 'ttcrpy':        
            if not ttcrpy_installed:
                raise ImportError('ttcrpy is not installed.')
        
            return self._forward_ttcrpy(
                sources, receivers, return_rays=return_rays, return_tt_field=return_tt_field, **kwargs)
        
    def plot_velocity_model(self, ax=None, colorbar=None, **kwargs):
        
        if ax is None:
            fig, ax = plt.subplots()
        
        if kwargs.get('cmap') is None: kwargs['cmap'] = 'Greys'
        if kwargs.get('alpha') is None: kwargs['alpha'] = 0.5
        
        cbar = ax.imshow(
            self.velocity_model.T,
            extent=[self.x[0], self.x[-1], self.z[-1], self.z[0]],
            **kwargs)
        
        if cbar is not None:
            if colorbar is True:
                plt.colorbar(cbar, ax=ax, label='Velocity [m/s]', shrink=0.5)
            elif type(colorbar) == dict:
                plt.colorbar(cbar, ax=ax, **colorbar)
        
        ax.set_xlim(self.x[0], self.x[-1])
        ax.set_ylim(self.z[-1], self.z[0])
        
        return ax
    
    def plot_receiver_locations(self, receivers, ax=None, **kwargs):
        
        if ax is None:
            fig, ax = plt.subplots()
            
        if kwargs.get('marker') is None: kwargs['marker'] = 10
        if kwargs.get('color') is None: kwargs['color'] = 'r'
        if kwargs.get('s') is None: kwargs['s'] = 100
        kwargs['clip_on'] = False
        
        ax.scatter(receivers[:, 0], receivers[:, 1], **kwargs)
        
        return ax
    
    def plot_source_locations(self, sources, ax=None, **kwargs):
        
        if ax is None:
            fig, ax = plt.subplots() 
        
        if kwargs.get('marker') is None: kwargs['marker'] = '*'
        if kwargs.get('color') is None: kwargs['color'] = 'k'
        if kwargs.get('s') is None: kwargs['s'] = 100
            
        kwargs['clip_on'] = False
        
        ax.scatter(sources[:, 0], sources[:, 1], **kwargs)
        
        return ax
    
    def plot_rays(
        self,
        rays=None,
        sources=None,
        receivers=None, 
        ax=None,
        **kwargs):
        
        if ax is None:
            fig, ax = plt.subplots()
        
        if rays is None and (sources is None or receivers is None):
            raise ValueError('Either rays or sources and receivers must be provided.')
        if rays is not None and (sources is not None or receivers is not None):
            raise ValueError('Either rays or sources and receivers must be provided.')            
        
        if sources is not None and receivers is not None:
            _, rays = self.calculate_tt(sources, receivers, return_rays=True,)

        if type(rays) != list:
            rays = [[rays]]
        if type(rays[0]) != list:
            rays = [rays]    
        
        if kwargs.get('color') is None: kwargs['color'] = 'k'
        if kwargs.get('linewidth') is None: kwargs['linewidth'] = 0.5
        if kwargs.get('alpha') is None: kwargs['alpha'] = 0.5

        for ray_i in rays:
            for ray_i_j in ray_i:
                ax.plot(ray_i_j[:, 0], ray_i_j[:, 1], **kwargs)
                kwargs['label'] = None
                                    
        return ax
    
    def plot_tt_field(
        self,
        tt_field=None,
        sources=None,
        ax=None,
        **kwargs):
        
        if ax is None:
            fig, ax = plt.subplots()
                
        if tt_field is None and sources is None:
            raise ValueError('Either tt_field or sources must be provided.')
        
        if tt_field is None and sources is not None:
            dummy_receivers = np.array([self.x[0], 0, self.z[0]])[None, :]
            _, tt_field = self.calculate_tt(sources, dummy_receivers, return_tt_field=True,)
            
        if type(tt_field) != list:
            tt_field = [tt_field]
            
        if kwargs.get('cmap') is None: kwargs['cmap'] = 'viridis'
        if kwargs.get('alpha') is None: kwargs['alpha'] = 0.5
            
        for tt_field_i in tt_field:
            ax.contour(self.x, self.z, tt_field_i.values[:, 0, :].T, extent=[self.x[0], self.x[-1], self.z[-1], self.z[0]], **kwargs)
            kwargs['label'] = None
        
        return ax
        

def velocity_grid_constructor(x, z, method='gradient', grid=True, **kwargs):
    if method == 'gradient':
        
        N_x = x.shape[0]
        N_z = z.shape[0]
        
        coarsness = kwargs.get('coarsness', 1)
        
        if N_z % coarsness != 0:
            raise ValueError('N_z must be a multiple of N_steps')
        else:
            N_c = N_z // coarsness
        
        vel = np.empty((N_c,))

        v_t  = kwargs['v_top']
        v_b  = kwargs['v_bottom'] 
        v_step = (v_t - v_b) / (N_c - 1)

        for n in range(N_c):
            vel[n] = v_b + n * v_step
            
        if coarsness > 1:
            vel = np.repeat(vel, coarsness)
            
        vel = np.tile(vel, N_x).reshape(N_x, N_z)

        return vel

    elif method == 'array':
        if 'velocity' in kwargs:
            return 1/kwargs['velocity']
        if 'slowness' in kwargs:
            return kwargs['slowness']
        else:
            raise ValueError('No velocity array provided')    
    else:
        raise ValueError('Unknown method')
        
    