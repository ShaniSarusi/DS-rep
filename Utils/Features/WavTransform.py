# -*- coding: utf-8 -*-
"""
Created on Sun Jan 08 09:15:37 2017

@author: awagner

This file contaion wavelet featre extraction
You can read more in the paper "Clustering functional data using wavelets"
"""

from scipy.interpolate import interpolate
import pywt
from future.utils import lmap
import numpy as np


class WavTransform():

    def __init__(self):
        print("Hello")

    def toDWT(self, sig):
        """
        Wavelet transform
        
        Input: 
            sig (1D numpy) - time signal

        Output: 
            ywav (1D numpy) - Wavelet transformation after interpolation to make the signal length as exponent of 2
        """
        x = np.arange(0, len(sig))
        f = interpolate.interp1d(x, sig)
        xnew = np.arange(0, len(sig)-1, float(len(sig)-1)/2**np.ceil(np.log2(len(sig))))
        ynew = f(xnew)
        ywav = pywt.wavedec(ynew - np.mean(ynew), pywt.Wavelet('db1'), mode='smooth')
        return ywav

    def contrib(self, x, rel=False):
        """
        Features as described in "Clustering functional data using wavelets"
        
        Input:
            x (1D numpy)- Signal in frequency domain (Wavelet transform)
            rel: if true we get relative features, else contributions features
        OutPut: 
            res(1D numpy) - Features as described in "Clustering functional data using wavelets"
        """
        J = len(x)
        res = np.zeros(J)
        for j in range(J):
            res[j] = np.sqrt(np.sum(x[j]**2))
        if rel is True:
            res = res/np.sum(res + 10**(-10))
            res = np.log(float(1)/(1-res))
        return res

    def createWavFeatures(self, LargeData):
        """
        Features as described in "Clustering functional data using wavelets"
        
        Input: 
            LargeData (numpy matrix) - every row is time signal
        Output: 
            contData (numpy matrix)- contributions features for each row
            relData (numpy matrix)- relative features for each row
        """
        print("Doing toDWT")
        WavData = lmap(self.toDWT, LargeData)
        print("relative wavelet")
        relData = lmap(lambda x: self.contrib(x, rel=True), WavData)
        print("cont wavelet")
        contData = lmap(lambda x: self.contrib(x, rel=False), WavData)
        return(np.column_stack((contData, relData)))