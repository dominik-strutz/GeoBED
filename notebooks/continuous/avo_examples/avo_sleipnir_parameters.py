import torch
import pandas as pd

import torch.distributions as dist

a_1_mean = torch.tensor([2270.0]) # m/s from Ghosh2020
a_1_std = torch.tensor([10.0])    # m/s filler value

b_1_mean = torch.tensor([854.0])  # m/s from Ghosh2020
b_1_std  = torch.tensor([10.0])   # m/s filler value

p_1_mean = torch.tensor([2100.0]) # kg/m3 from Ghosh2020
p_1_std  = torch.tensor([10.0])   # kg/m3 filler value

d_mean = torch.tensor([1000.0])   # m from Dupuy2017
d_std  = torch.tensor([50.0])     # m filler value

# Grain properties
K_grain_mean = torch.tensor([39.3])*1e9              # Pa from Ghosh2020
K_grain_std  = torch.tensor([3.6/100])*K_grain_mean  # Pa from Ghosh2020

# G_grain_mean = torch.tensor([44.8])*1e9              # Pa from Ghosh2020
# G_grain_std  = torch.tensor([1.8/100])*G_grain_mean  # Pa from Ghosh2020

rho_grain_mean = torch.tensor([2664.0])                  # kg/m3 from Ghosh2020
rho_grain_std  = torch.tensor([0.1/100])*rho_grain_mean  # kg/m3 from Ghosh2020


# Frame properties
K_frame_mean = torch.tensor([2.56])*1e9                 # Pa from Ghosh2020
K_frame_std  = torch.tensor([3.0/100])*K_frame_mean     # Pa from Ghosh2020

G_frame_mean = torch.tensor([8.5])*1e9                  # Pa from Ghosh2020
G_frame_std  = torch.tensor([3.0/100])*G_frame_mean     # Pa from Ghosh2020

# k_frame_mean = torch.tensor([2e-12])                    # m2 from Ghosh2020
# k_frame_std  = torch.tensor([75.0/100])*k_frame_mean    # m2 from Ghosh2020

# m_frame_mean = torch.tensor([1.0])                      # from Ghosh2020
# m_frame_std  = torch.tensor([25.0/100])*m_frame_mean    # from Ghosh2020

porosity_mean = torch.tensor([0.37])                    # from Ghosh2020
porosity_std  = torch.tensor([6.75/100])*porosity_mean  # from Ghosh2020

# Brine properties
K_brine_mean = torch.tensor([2.35])*1e9                  # Pa from Ghosh2020
K_brine_std  = torch.tensor([3.24/100])*K_brine_mean     # Pa from Ghosh2020

rho_brine_mean = torch.tensor([1030.0])                  # kg/m3 from Ghosh2020
rho_brine_std  = torch.tensor([1.94/100])*rho_brine_mean # kg/m3 from Ghosh2020

# mu_brine_mean = torch.tensor([0.00069])                  # Pa*s from Ghosh2020
# mu_brine_std  = torch.tensor([1.5/100])*mu_brine_mean    # Pa*s from Ghosh2020

# CO2 properties

K_co2_mean = torch.tensor([0.08])*1e9                    # Pa from Ghosh2020
K_co2_std  = torch.tensor([53.0/100])*K_co2_mean         # Pa from Ghosh2020

rho_co2_mean = torch.tensor([700.0])                     # kg/m3 from Ghosh2020
rho_co2_std  = torch.tensor([11.0/100])*rho_co2_mean     # kg/m3 from Ghosh2020

# mu_co2_mean = torch.tensor([0.000006])                   # Pa*s from Ghosh2020
# mu_co2_std  = torch.tensor([17.0/100])*mu_co2_mean       # Pa*s from Ghosh2020

nuisance_parameter_table = pd.DataFrame(
    data=[
        [a_1_mean.item(), b_1_mean.item(), p_1_mean.item(), d_mean.item(), K_grain_mean.item(), rho_grain_mean.item(), K_frame_mean.item(), G_frame_mean.item(), porosity_mean.item(), K_brine_mean.item(), rho_brine_mean.item(), K_co2_mean.item(), rho_co2_mean.item()],
        [a_1_std.item(), b_1_std.item(), p_1_std.item(), d_std.item(), K_grain_std.item(), rho_grain_std.item(), K_frame_std.item(), G_frame_std.item(), porosity_std.item(), K_brine_std.item(), rho_brine_std.item(), K_co2_std.item(), rho_co2_std.item()]
    ],
    index=['mean', 'std'],
    columns=[r'$\alpha_1$', r'$\beta_1$', r'$\rho_1$', r'$d$', r'$K_{grain}$', r'$\rho_{grain}$', r'$K_{frame}$', r'$G_{frame}$', r'$\phi$', r'$K_{brine}$', r'$\rho_{brine}$', r'$K_{co2}$', r'$\rho_{co2}$']
)

name_to_index = {
    'alpha_1': 0,
    'beta_1': 1,
    'rho_1': 2,
    'd': 3,
    'K_grain': 4,
    'rho_grain': 5,
    'K_frame': 6,
    'G_frame': 7,
    'porosity': 8,
    'K_brine': 9,
    'rho_brine': 10,
    'K_co2': 11,
    'rho_co2': 12,
}
    

nuisance_dist = dist.Independent(
    dist.Normal(
        torch.cat([a_1_mean, b_1_mean, p_1_mean, d_mean, K_grain_mean, rho_grain_mean, K_frame_mean, G_frame_mean, porosity_mean, K_brine_mean, rho_brine_mean, K_co2_mean, rho_co2_mean,]),
        torch.cat([a_1_std, b_1_std, p_1_std, d_std, K_grain_std, rho_grain_std, K_frame_std, G_frame_std, porosity_std, K_brine_std, rho_brine_std, K_co2_std, rho_co2_std])
    ),
    1)