# -*- coding: utf-8 -*-
"""
Created on Tue Aug 02 11:40:01 2016

Implementation of butterworth filter

@author: awagner
"""
# from spectrum import *
from scipy.signal import welch
import numpy as np


def spectogram_and_normalize(data):
    fdata = np.abs(welch(data-np.mean(data), 50, nperseg=len(data))[1])  # periodogram is a function in spectro something.   data-mean makes the average zero.
    fdata[range(3)] = 0     
    fdata_normalize = (fdata - np.min(fdata))/(np.max(fdata)-np.min(fdata))
    #p = probVec(fdata)   #probvec is likes softmax. makes everything sum to one. HOwerver, fourier trans is both sided, thus since we take one side, we need to multiply by two.
    return fdata_normalize




