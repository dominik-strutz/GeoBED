import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import pykonal
except ImportError:
    pass

class TTHelper():
    
    def __init__(self, coord_sys="cartesian") -> None:
        self.coord_sys = coord_sys
        self.model_set = True

    def set_model(self, x, dx, velocity_model,
                  y=np.array([0]), dy=1.0, 
                  z=np.array([0]), dz=1.0):
        self.x, self.y, self.z    = x, y, z
        self.min_coords           = min(x), min(y), min(z)
        self.node_intervals       = dx, dy, dz
        self.dx, self.dy, self.dz = dx, dy, dz
        self.npts                 = len(x), len(y), len(z)
        self.velocity_model       = velocity_model

        self.model_set = True

    def calculate_tt(self, src, rec_list):
        
        solver = pykonal.solver.PointSourceSolver(coord_sys=self.coord_sys)
        solver.velocity.min_coords     = self.min_coords
        solver.velocity.node_intervals = self.node_intervals
        
        solver.velocity.npts           = self.npts
        solver.velocity.values         = self.velocity_model[:, None, :]

        solver.src_loc                 = np.array( (src[0], 0, src[1]) ).astype('double')
                
        solver.solve()
        
        rec_list = np.vstack( [rec_list[:, 0], np.zeros_like(rec_list[:, 0]), rec_list[:, 1]] ).T

        tt_list = solver.traveltime.resample(rec_list.astype('float'))
        
        return tt_list

    def calculate_tt_diff(self, src, rec_list, vpvs_ratio=(np.sqrt(3) - 1)):
        
        tt_list = self.calculate_tt(src, rec_list)
        
        return tt_list * vpvs_ratio
    
    def plot_model(self, prior_realisations=None, receivers=None,
                   xlim=None, ylim=None, colorbar=False,
                   vmin=None, vmax=None, title=None, im_cmap='viridis',
                   plot_rays=None, ax=None, figsize=(16, 7)):
        
        if self.model_set:
                
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize, dpi=200)
            
            im = ax.imshow(self.velocity_model.T,
                            extent=(min(self.x), max(self.x),
                                    max(self.z), min(self.z)),
                            origin='upper', cmap=im_cmap,
                            vmin=vmin,
                            vmax=vmax,
                            aspect='auto',
                            zorder=0)
            
            if receivers is not None:
                ax.scatter(receivers[:,0], receivers[:,2], marker=10, s=50, color='r', zorder=10, clip_on=False)
            
            if prior_realisations is not None:
                
                ax.scatter(prior_realisations[:,0], prior_realisations[:,1], marker='+', color='r', linewidths=1, alpha=0.1, zorder=100)
                ax.set_xlim(min(self.x), max(self.x))
                ax.set_ylim(max(self.z), min(self.z))
            
            if plot_rays is not None:
                solver = pykonal.solver.PointSourceSolver(coord_sys=self.coord_sys)
                solver.velocity.min_coords     = self.min_coords
                solver.velocity.node_intervals = self.node_intervals
                solver.velocity.npts           = self.npts
                solver.velocity.values         = self.velocity_model[:, None, :]

                solver.src_loc=np.array( plot_rays ).astype('double')
                
                solver.solve()
            
                for rec in receivers:
                    ray_coords = solver.traveltime.trace_ray( np.array( rec, dtype=float))
                    ax.plot(ray_coords[:, 0], ray_coords[:, 2], 'k', alpha=0.4)
            
            ax.set_xlabel('x [km]')
            ax.set_ylabel('z [km]')
            
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)
            if colorbar:
                cbar = plt.colorbar(im)
                cbar.set_label('Wavespeed [km/s]')
            if title:
                ax.set_title(title)
            
            if ax is None:
                plt.show()
            else:
                return ax
                    
                    
                    
            