# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:36:46 2017

@author: imazeh
"""

import numpy as np
from future.utils import lmap


def project_gravity(x, y, z, num_samples_per_interval=None, round_up_or_down='down', return_only_vertical=False):
    """
    Projection of 3D time signal to 2D
    
    Input:
        x (1D numpy) - time signal X samples
        y (1D numpy) - time signal Y samples
        z (1D numpy) - time signal Z samples
        num_samples_per_interval (integer) - cut the signal to num_samples_per_interval sub intervals
                                     and preform the 2D projection on the sub intervals
        round_up_or_down (string, down or up) - length(x)/num_samples_per_interval should 
                                                be ceil or floor
        return_only_vertical (boolean) - If True return only vertical axis
    
    Output:
        v (1D numpy) - vertical projection
        h (1D numpy) - horizontal projection
                                  
    """
    if num_samples_per_interval is None:
        v, h = project_gravity_xyz(x, y, z)
        if return_only_vertical:
            return v
        else:
            return v, h

    # set number of intervals
    n = len(x)/num_samples_per_interval
    if round_up_or_down == 'down':
        n = np.floor(n).astype(int)
        n = np.max([1, n])
    elif round_up_or_down == 'up':
        n = np.ceil(n).astype(int)

    # set window size
    win_size = np.floor(len(x)/n).astype(int)

    # perform sliding windows
    idx_start = 0
    v = []
    h = []

    # TODO Chunk the samples below evenly. Do this by dividing len(x) each time rather than the current implementation
    for i in range(n):
        idx_start = i * win_size
        idx_end = (i + 1) * win_size
        if i == n-1:  # last iteration
            idx_end = -1
        x_i = x[idx_start:idx_end]
        y_i = y[idx_start:idx_end]
        z_i = z[idx_start:idx_end]
        ver_i, hor_i = project_gravity_xyz(x_i, y_i, z_i)
        v.append(ver_i)
        h.append(hor_i)
    if return_only_vertical:
        return np.hstack(v)
    return np.hstack(v), np.hstack(h)


def project_gravity_xyz(x, y, z):
    xyz = np.stack((x, y, z), axis=1)
    return project_gravity_core(xyz)


def project_gravity_core(xyz):
    """
    Prjection of data set to 2 dim
    
    Input:
        xyz (3d numy array) - 0 dimention is number os samples, 1 dimenstion is length of signals 
                    and 2 dim is number of axis
                    
    Output:
        ver (1d numpy) - Vertical projection
        hor (1d numpy) - Horizontal projection
        
    """
    ver = []
    hor = []
    
    # mean for each axis
    G = [np.mean(xyz[:, 0]), np.mean(xyz[:, 1]), np.mean(xyz[:, 2])]
    G_norm = G/np.sqrt(sum(np.power(G, 2)))
    
    # The projection is here
    for i in range(len(xyz[:, 0])):
        ver.append(np.dot([xyz[i, :]], G))
        hor.append(np.sqrt(np.dot(xyz[i, :]-ver[i]*G_norm, xyz[i, :]-ver[i]*G_norm)))
        
    ver = np.reshape(np.asarray(ver), len(ver))
    return np.asarray(ver), np.asarray(hor)


def project_from_3_to_2_dims(x, y, z, interval_length=None):
    """
    Input:
        x - x axis numpy array, every raw is sample
        y - y axis numpy array, every raw is sample
        z - z axis numpy array, every raw is sample
        interval_length - interval_length is the length of each interval, in number of samples
        
    Ouput:
        ver_proj - vertical projection
        hor_proj - horizontal projection
    """
    HR = lmap((lambda x_lam, y_lam, z_lam: project_gravity(x_lam, y_lam, z_lam, interval_length)), x, y, z)
    ver_proj = np.asarray([i[0] for i in HR])
    hor_proj = np.asarray([i[1] for i in HR])
    return ver_proj, hor_proj



