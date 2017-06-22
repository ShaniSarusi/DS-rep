# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:36:46 2017

@author: imazeh
"""

import numpy as np
from future.utils import lmap


def project_from_3_to_2_dims(x, y, z):
    """
    Input:
        x - x axis numpy array, every raw is sample
        y - y axis numpy array, every raw is sample
        z - z axis numpy array, every raw is sample
    Ouput:
        ver_proj - vertical projection
        hor_proj - horizantel projection
    """
    XYZ = np.stack((x, y, z), axis=2)
    XYZ = np.reshape(XYZ, (np.shape(XYZ)[0], np.shape(XYZ)[1]*np.shape(XYZ)[2]))
    HR = lmap(projGrav, XYZ)
    ver_proj = np.asarray([i[0] for i in HR])
    hor_proj = np.asarray([i[1] for i in HR])
    return ver_proj, hor_proj


def projGrav(XYZ):
    """
    Input:
        XYZ - a 2d array, each row is XYZ = np.stack((x, y, z), axis=2)
        where x, y , z are axis numpy array, every raw is sample
    Ouput:
        ver_proj - vertical projection
        hor_proj - horizantel projection
    """
    XYZ = np.reshape(XYZ, (int(len(XYZ)/3), 3))
    ver = []
    hor = []
    G = [np.mean(XYZ[:, 0]), np.mean(XYZ[:, 1]), np.mean(XYZ[:, 2])]
    G_norm = G/np.sqrt(sum(np.power(G, 2)))
    for i in range(len(XYZ[:, 0])):
        ver.append(np.dot([XYZ[i, :]], G))
        hor.append(np.sqrt(np.dot(XYZ[i, :]-ver[i]*G_norm, XYZ[i, :]-ver[i]*G_norm)))
    ver = np.reshape(np.asarray(ver), len(ver))
    return np.asarray(ver), np.asarray(hor)


def projGrav_one_samp(x, y, z):
    """
    Input:
        x - time signal
        y - time singal
        z - time signal
    Output:
        ver - vertical axis
        hor - horizantal axis
    """
    XYZ = np.stack((x, y, z), axis=1)
    ver = []
    hor = []
    G = [np.mean(XYZ[:, 0]), np.mean(XYZ[:, 1]), np.mean(XYZ[:, 2])]
    G_norm = G/np.sqrt(sum(np.power(G, 2)))
    for i in range(len(XYZ[:, 0])):
        ver.append(np.dot([XYZ[i, :]], G))
        hor.append(np.sqrt(np.dot(XYZ[i, :]-ver[i]*G_norm, XYZ[i, :]-ver[i]*G_norm)))
    ver = np.reshape(np.asarray(ver), len(ver))
    return np.asarray(ver), np.asarray(hor)


def project_gravity(x, y, z, num_samples_per_interval=None, round_up_or_down='down'):
    if num_samples_per_interval is None:
        return projGrav_one_samp(x, y, z)

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
        ver_i, hor_i = projGrav_one_samp(x_i, y_i, z_i)
        v.append(ver_i)
        h.append(hor_i)
    return np.hstack(v), np.hstack(h)


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    print(avg)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out
