# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:36:46 2017

@author: imazeh
"""

import numpy as np
from future.utils import lmap


def project_gravity(x, y, z, num_samples_per_interval=None, round_up_or_down='down'):
    if num_samples_per_interval is None:
        return project_gravity_xyz(x, y, z)

    # set number of intervals
    n = len(x)/num_samples_per_interval
    if round_up_or_down == 'down':
        n = np.floor(n).astype(int)
    elif round_up_or_down == 'up':
        n = np.ceil(n).astype(int)

    # set window size
    win_size = np.floor(len(x)/n).astype(int)

    # perform sliding windows
    idx_start = 0
    v = []
    h = []
    for i in range(n):  # TODO - chunk the samples more evenly by dividing len(x) each time
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
    return np.hstack(v), np.hstack(h)


def project_gravity_xyz(x, y, z):
    xyz = np.stack((x, y, z), axis=1)
    return project_gravity_core(xyz)


def project_gravity_core(xyz):
    ver = []
    hor = []
    G = [np.mean(xyz[:, 0]), np.mean(xyz[:, 1]), np.mean(xyz[:, 2])]
    G_norm = G/np.sqrt(sum(np.power(G, 2)))
    for i in range(len(xyz[:, 0])):
        ver.append(np.dot([xyz[i, :]], G))
        hor.append(np.sqrt(np.dot(xyz[i, :]-ver[i]*G_norm, xyz[i, :]-ver[i]*G_norm)))
    ver = np.reshape(np.asarray(ver), len(ver))
    return np.asarray(ver), np.asarray(hor)


def project_from_3_to_2_dims(x, y, z):
    """
    Input:
        x - x axis numpy array, every raw is sample
        y - y axis numpy array, every raw is sample
        z - z axis numpy array, every raw is sample
    Ouput:
        ver_proj - vertical projection
        hor_proj - horizontal projection
    """
    HR = lmap((lambda x_lam, y_lam, z_lam: project_gravity(x_lam, y_lam, z_lam)), x, y, z)
    ver_proj = np.asarray([i[0] for i in HR])
    hor_proj = np.asarray([i[1] for i in HR])
    return ver_proj, hor_proj



