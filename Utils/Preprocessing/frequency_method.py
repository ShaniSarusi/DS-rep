# -*- coding: utf-8 -*-
"""
Created on Tue Aug 02 11:40:01 2016

Implementation of butterworth filter

@author: awagner
"""
# from spectrum import *
from scipy.signal import welch
import numpy as np


def spectogram_and_normalize(data,normlisedata = False):
    # periodogram is a function in spectro something.  
    #data-mean makes the average zero.
    fdata = np.abs(welch(data-np.mean(data), 50, nperseg=len(data))[1])  
    
    if(normlisedata ==True):
        fdata = (fdata - np.min(fdata))/(np.max(fdata)-np.min(fdata))
    return fdata




