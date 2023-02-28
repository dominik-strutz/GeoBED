import torch

def calculate_avo(x, a_1, a_2, d):
    '''Just a simple helper function to calculate the AVO function.'''
    return calculate_avo_Guest(x, a_1, a_2, d)

def calculate_avo_Berg(x, a_1, a_2, d):
    """AVO function for interface without density difference and a=sqrt(3)*beta.

    Args:
        x (torch.tensor): offset tensor
        a_1 (torch.tensor): a in upper layer
        a_2 (torch.tensor): a in lower layer
        d (torch.tensor): depth of interface

    Returns:
        torch.tensor: tensor of refelection coefficient
    """    
    
    c = 1 / torch.sqrt(torch.tensor(3.0))
    
    x = torch.complex(x, torch.zeros(x.shape))
    
    i_1 = torch.atan( x / (2 * d) )

    i_2 = torch.asin( a_2 / a_1 * torch.sin(i_1) )
    i = 0.5 * (i_1 + i_2)

    R_p = torch.abs(
        ( 0.5 * ( 1.0 + torch.tan(i).pow(2) ) - 4.0 * c** 2* torch.sin(i).pow(2) ) * (a_2 - a_1) / ( (a_1 + a_2)*0.5)
        )

    return R_p

def calculate_avo_Guest(x, a_1, a_2, d):
    """AVO function for interface without density difference and a=sqrt(3)*beta.

    Args:
        x (torch.tensor): offset tensor
        a_1 (torch.tensor): a in upper layer
        a_2 (torch.tensor): a in lower layer
        d (torch.tensor): depth of interface

    Returns:
        torch.tensor: tensor of refelection coefficient
    """ 
    
    theta_1 = torch.atan( x / (2 * d) )
    
    c = 1 / torch.sqrt(torch.tensor(3.0))
    a_1 = a_1 * torch.ones_like(a_2)
    b_1 = c*a_1
    b_2 = c*a_2
    r_1 = 1000.0 * torch.ones_like(a_2)
    r_2 = 1000.0 * torch.ones_like(a_2)
        
    result = zoeppritz_solver_rpp(theta_1, a_1, a_2, b_1, b_2, r_1, r_2)

    return result

def zoeppritz_solver(theta_1, a_1, a_2, b_1, b_2, r_1, r_2):
    """Solver for Zoeppritz equation (see: https://en.wikipedia.org/wiki/Zoeppritz_equations). """
    
    theta_1 = torch.complex(theta_1, torch.zeros(theta_1.shape))

    p = torch.sin(theta_1) / a_1 
    theta_2 = torch.arcsin(p * a_2)
    phi_1 = torch.arcsin(p * b_1)
    phi_2 = torch.arcsin(p * b_2)
    
    B = torch.vstack([torch.sin(theta_1), torch.cos(theta_1), torch.sin(2*theta_1), torch.cos(2*theta_1)]).T
        
    M = torch.vstack([-torch.sin(theta_1), -torch.cos(phi_1), torch.sin(theta_2),  torch.cos(phi_2),
                       torch.cos(theta_1), -torch.sin(phi_1), torch.cos(theta_2), -torch.sin(phi_2),
                    
                      torch.sin(2*phi_1), a_1/b_1 * torch.cos(2*phi_1),
                       (r_2*b_2**2*a_1)/(r_1*b_1**2*a_2) * torch.cos(2*phi_1),
                       (r_2*b_2*a_1)/(r_1*b_1**2) * torch.cos(2*phi_2), 
                       
                      -torch.cos(2*phi_1), b_1/a_1 * torch.sin(2*phi_1),
                       (r_2*a_2)/(r_1*a_1)*torch.cos(2*phi_2),
                       (r_2*b_2)/(r_1*a_1)*torch.sin(2*phi_2)]).T
    
    M = M.reshape(theta_1.shape + (4,4))
    
    result = torch.linalg.solve(M, B)
    result = torch.abs(result)
        
    return result

def zoeppritz_solver_rpp(theta_1, a_1, a_2, b_1, b_2, r_1, r_2):
    """"Taken from https://github.com/agilescientific/bruges/blob/main/bruges/reflection/reflection.py"""
    
    theta_1 = torch.complex(theta_1, torch.zeros(theta_1.shape))

    p = torch.sin(theta_1) / a_1  # Ray parameter
    theta2 = torch.arcsin(p * a_2)
    phi1 = torch.arcsin(p * b_1)  # Reflected S
    phi2 = torch.arcsin(p * b_2)  # Transmitted S

    a = r_2 * (1 - 2 * torch.sin(phi2)**2.) - r_1 * (1 - 2 * torch.sin(phi1)**2.)
    b = r_2 * (1 - 2 * torch.sin(phi2)**2.) + 2 * r_1 * torch.sin(phi1)**2.
    c = r_1 * (1 - 2 * torch.sin(phi1)**2.) + 2 * r_2 * torch.sin(phi2)**2.
    d = 2 * (r_2 * b_2**2 - r_1 * b_1**2)

    E = (b * torch.cos(theta_1) / a_1) + (c * torch.cos(theta2) / a_2)
    F = (b * torch.cos(phi1) / b_1) + (c * torch.cos(phi2) / b_2)
    G = a - d * torch.cos(theta_1)/a_1 * torch.cos(phi2)/b_2
    H = a - d * torch.cos(theta2)/a_2 * torch.cos(phi1)/b_1

    D = E*F + G*H*p**2

    rpp = torch.abs((1/D) * (F*(b*(torch.cos(theta_1)/a_1) - c*(torch.cos(theta2)/a_2)) \
                   - H*p**2 * (a + d*(torch.cos(theta_1)/a_1)*(torch.cos(phi2)/b_2))))

    return rpp
    