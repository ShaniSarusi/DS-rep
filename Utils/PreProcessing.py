import copy
import scipy.signal as sig
import numpy as np


def butter_filter_lowpass(data, order, sampling_rate, freq):
    nyq = 0.5 * sampling_rate
    b, a = sig.butter(order, float(freq)/float(nyq), btype='lowpass')
    return sig.filtfilt(b, a, copy.deepcopy(data))


def butter_filter_highpass(data, order, sampling_rate, freq):
    nyq = 0.5 * sampling_rate
    b, a = sig.butter(order, float(freq)/float(nyq), btype='highpass')
    return sig.filtfilt(b, a, copy.deepcopy(data))


def vc_axis():
    pass


# Projection of the data to 2 dimension
def projGrav(XYZ):
    #XYZ = np.reshape(XYZ,(int(len(XYZ)/3),3))
    ver = []; hor = []
    G = [np.mean(XYZ[:,0]),np.mean(XYZ[:,1]),np.mean(XYZ[:,2])]
    G_norm = G/np.sqrt(sum(np.power(G,2)))
    for i in range(len(XYZ[:,0])):
        ver.append(np.dot([XYZ[i,:]],G))
        hor.append(np.sqrt(np.dot(XYZ[i,:]-ver[i]*G_norm,XYZ[i,:]-ver[i]*G_norm)))
    ver = np.reshape(np.asarray(ver),len(ver))
    return np.asarray(ver), np.asarray(hor)