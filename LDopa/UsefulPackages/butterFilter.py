# -*- coding: utf-8 -*-
"""
Created on Tue Aug 02 11:40:01 2016

Implimantion of butterworth filter

@author: awagner
"""
#from spectrum import *
from scipy.signal import welch
import numpy as np


def probVec(data):
    x = abs(data)/float(np.sum(abs(data)))
    return x

def absfft(data):
    fdata = np.abs(welch(data-np.mean(data),50, nperseg = 250)[1])  # periodogram is a function in spectro something.   data-mean makes the average zero.
    fdata[range(3)] = 0     
    p =  (fdata - np.min(fdata))/(np.max(fdata)-np.min(fdata))
    #p = probVec(fdata)   #probvec is likes softmax. makes everything sum to one. HOwerver, fourier trans is both sided, thus since we take one side, we need to multiply by two.
    return p

def wavtrans(data):
    (cA, y) = pywt.dwt(np.lib.pad(hor[0], (1,1), 'constant', constant_values=(0, 0)), 'db1')
    y = (y - np.min(y))/(np.max(y)-np.min(y))
    return y




