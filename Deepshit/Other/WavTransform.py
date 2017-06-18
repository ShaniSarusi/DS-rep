# -*- coding: utf-8 -*-
"""
Created on Sun Jan 08 09:15:37 2017

@author: awagner
"""

from scipy.interpolate import interpolate
import pywt
from scipy import signal
import pandas as pd
from future.utils import lmap
import numpy as np

class wavtransform():
    
    def __init__(self):
        print("Hello")
        
    def toDWT(self,sig):
       x = np.arange(0, len(sig))
       f = interpolate.interp1d(x, sig)
       xnew = np.arange(0,len(sig)-1,float(len(sig)-1)/2**np.ceil(np.log2(len(sig))))
       ynew = f(xnew)
       ywav = pywt.wavedec(ynew - np.mean(ynew) , pywt.Wavelet('db1'),mode = 'smooth')
       return ywav
        
    def contrib(self,x,rel=False):
       J  = len(x)
       res = np.zeros(J)
       for j in range(J):
           res[j] = np.sqrt(np.sum(x[j]**2))        
       if rel == True:
           res = res/np.sum(res)
           res = np.log(float(1)/(1-res))       
       return res
    
    
    def createWavFeatures(self,LargeData):
        print("Doing toDWT")
        verWav = lmap(self.toDWT, LargeData)
        print("relative wavelet")
        relData = lmap(lambda x:self.contrib(x,rel=True), verWav)
        print("cont wavelet")
        contData =lmap(lambda x:self.contrib(x,rel=False), verWav)
        return(np.column_stack((contData,relData)))
        
