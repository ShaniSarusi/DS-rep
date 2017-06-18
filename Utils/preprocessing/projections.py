# -*- coding: utf-8 -*-
"""
Created on Thu May 25 15:36:46 2017

@author: imazeh
"""

import numpy as np
from future.utils import lmap
import pywt

#    import sys
#    sys.path.insert(0, 'C:\\Users\\imazeh\\Itzik\\Health_prof\\git4\\DataScientists\\LDopa\\useful_packages\\')
#    from FunctionForPredWithDEEP import projGrav

"""
Input:
    x - x axis numpy array, every raw is sample
    y - y axis numpy array, every raw is sample
    z - z axis numpy array, every raw is sample
Ouput:
    ver_proj - vertical projection
    hor_proj - horizantel projection
"""
def project_from_3_to_2_dims(x, y, z):
    XYZ = np.stack((x, y, z), axis=2)
    XYZ = np.reshape(XYZ, (np.shape(XYZ)[0], np.shape(XYZ)[1]*np.shape(XYZ)[2]))
    HR = lmap(projGrav, XYZ)
    ver_proj = np.asarray([i[0] for i in HR])
    hor_proj = np.asarray([i[1] for i in HR])
    return ver_proj, hor_proj

"""
Input:
    XYZ - a 2d array, each row is XYZ = np.stack((x, y, z), axis=2) 
          where x, y , z are axis numpy array, every raw is sample
Ouput:
    ver_proj - vertical projection
    hor_proj - horizantel projection
"""
def projGrav(XYZ):
    XYZ = np.reshape(XYZ,(int(len(XYZ)/3),3))
    ver = []; hor = []
    G = [np.mean(XYZ[:,0]),np.mean(XYZ[:,1]),np.mean(XYZ[:,2])]
    G_norm = G/np.sqrt(sum(np.power(G,2)))
    for i in range(len(XYZ[:,0])):
        ver.append(np.dot([XYZ[i,:]],G))
        hor.append(np.sqrt(np.dot(XYZ[i,:]-ver[i]*G_norm,XYZ[i,:]-ver[i]*G_norm)))
    ver = np.reshape(np.asarray(ver),len(ver))
    return np.asarray(ver), np.asarray(hor)

"""
Input:
    x - time signal
    y - time singal
    z - time signal
Output:
    ver - vertical axis
    hor - horizantal axis
"""
def projGrav_one_samp(x,y,z):
    XYZ = np.stack((x, y, z), axis=1)
    ver = []; hor = []
    G = [np.mean(XYZ[:,0]),np.mean(XYZ[:,1]),np.mean(XYZ[:,2])]
    G_norm = G/np.sqrt(sum(np.power(G,2)))
    for i in range(len(XYZ[:,0])):
        ver.append(np.dot([XYZ[i,:]],G))
        hor.append(np.sqrt(np.dot(XYZ[i,:]-ver[i]*G_norm,XYZ[i,:]-ver[i]*G_norm)))
    ver = np.reshape(np.asarray(ver),len(ver))
    return np.asarray(ver), np.asarray(hor)


def projGrav_one_samp_try(x,y,z,num_of_interval = 1):
    List_x = chunkIt(x, num_of_interval)
    List_y = chunkIt(y, num_of_interval)
    List_z = chunkIt(z, num_of_interval)
    hor_per_interval = []; ver_per_interval = [];
    for index_list in range(len(List_x)):       
        print(index_list)
        XYZ = np.stack((List_x[index_list], List_y[index_list], List_z[index_list]), axis=1)
        ver = []; hor = []     
        G = [np.mean(XYZ[:,0]),np.mean(XYZ[:,1]),np.mean(XYZ[:,2])]
        G_norm = G/np.sqrt(sum(np.power(G,2)))
        for i in range(len(XYZ[:,0])):
            ver.append(np.dot([XYZ[i,:]],G))
            hor.append(np.sqrt(np.dot(XYZ[i,:]-ver[i]*G_norm,XYZ[i,:]-ver[i]*G_norm)))
        ver = np.reshape(np.asarray(ver),len(ver))
        ver_per_interval.append(np.asarray(ver))
        hor_per_interval.append(np.asarray(hor))
    ver_output = np.hstack(ver_per_interval)
    hor_output = np.hstack(hor_per_interval)
    return ver_output, hor_output


def chunkIt(seq, num):
  avg = len(seq) / float(num)
  print(avg)
  out = []
  last = 0.0

  while last < len(seq):
    out.append(seq[int(last):int(last + avg)])
    last += avg

  return out








