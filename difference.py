import numpy as np

def d_dx(var, dx):
    '''
    calculate the partial x difference for given dx (dx varies to y).

    Parameters
    -----
    var: a two dimensional numpy array
    dx: one dimensional numpy array with len = var.shape[0]
    
    Returns
    -----
    dvar_dx: var differential to x. the shape is as same as var.shape
    '''
    dvar_dx = np.zeros(var.shape)

    # forward finite difference for the start points
    dvar_dx[:, 0] = (var[:, 1] - var[:, 0]) / dx
    
    # backward finite difference for the end points
    dvar_dx[:, -1] = (var[:, -1] - var[:, -2]) / dx
    
    # center finite difference for the other points
    for j in range(1, var.shape[1]-1):
        dvar_dx[:, j] = (var[:, j+1] - var[:, j-1]) / (2 * dx)
        
    return dvar_dx

def d_dy(var, dy):
    '''
    calculate the partial y difference for given dy.
    
    Parameters
    ------
    var: a two dimensional numpy array
    dy: int or float (scalar)
    
    Returns
    -----
    dvar_dy: var differential to y. the shape is as same as var.shape
    '''
    dvar_dy = np.zeros(var.shape)
    
    # forward finite difference for the start points
    dvar_dy[0, :] = (var[1, :] - var[0, :]) / dy
    
    # backward finite difference for the end points
    dvar_dy[-1, :] = (var[-1, :] - var[-2, :]) / dy
    
    # center finite difference for the other points
    for i in range(1, var.shape[0]-1):
        dvar_dy[i, :] = (var[i+1, :] - var[i-1, :]) / (2 * dy)
    
    return dvar_dy