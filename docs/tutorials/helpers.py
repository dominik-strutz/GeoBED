import numpy as np
import torch

def construct_covmat(theta, ratio, scaling): 
    theta = -np.radians(theta)
    ratio = ratio
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    S = scaling * np.diag([ratio, 1])
    L = S**2
    return torch.tensor(R@L@R.T, dtype=torch.float32)