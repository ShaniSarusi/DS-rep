# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 10:39:06 2017

@author: awagner
"""


import numpy as np
from pyspark import SQLContext
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.window import Window
from pyspark.sql.functions import ceil, unix_timestamp
from pyspark.sql.functions import rank
from pyspark.sql.functions import collect_list, array
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.sql.types import *

from scipy.interpolate import interpolate
import pywt
from future.utils import lmap
import numpy as np
from functools import partial


def parse(s):
    return eval(str(s))
parse_udf = udf(parse, ArrayType(FloatType(), False))

def merge_integers(raw1, raw2):
    return int(str(raw1) + str(raw2))
merge_integers_udf = udf(merge_integers)


def find_milisec(raw):
    return (raw)[(raw.find('.')+1):(raw.find('.')+3)]
find_milisec_udf = udf(find_milisec)


def give_my_key(*args):
    key = 0
    for i in args:
        key += float(i)
    return key

give_my_key_udf = udf(give_my_key)

sort_vec_udf = udf(lambda X, Y: [x for _,x in sorted(zip(Y,X))])


def slidind_window(axis, time_stamp, slide, window_size):
    #axis = eval(axis)
    t = time_stamp[0]
    windows = []
    windows.append(axis[:(window_size*50+1)])
    for time1 in range(len(time_stamp)):
        if float(time_stamp[time1]) >= float(t) + 100*slide:
            if time1+window_size*50 < len(time_stamp):
                windows.append(axis[time1:(time1+window_size*50+1)])
                t =  time_stamp[time1]
    
    return (windows)



def sliding_window(slide, window_size):
    sliding_window_new = partial(slidind_window, 
                                 slide = slide, window_size = window_size)
    return udf(sliding_window_new,
                         ArrayType(ArrayType(FloatType(), False), False))

def denoise(data):
    """
    Denoise the data with wavelet and
    Input:
        data - time signal
    Output:
        result - signal after denoising
    """
    data = data - np.mean(data) + 0.1
    WC = pywt.wavedec(data, 'sym8')
    threshold = 0.01*np.sqrt(2*np.log2(256))
    NWC = lmap(lambda x: pywt.threshold(x, threshold, 'soft'), WC)
    result = pywt.waverec(NWC, 'sym8')
    return  Vectors.dense(result)

denoise_func = udf(denoise,  VectorUDT())



def toDWT(sig, rel = False):

        x = np.arange(0, len(sig))
        f = interpolate.interp1d(x, sig)
        xnew = np.arange(0, len(sig)-1, float(len(sig)-1)/2**np.ceil(np.log2(len(sig))))
        ynew = f(xnew)
        x = pywt.wavedec(ynew - np.mean(ynew), pywt.Wavelet('db1'), mode='smooth')
                
        J = len(x)
        res = np.zeros(J)
        for j in range(J):
            res[j] = float(np.sqrt(np.sum(x[j]**2)))
        if rel is True:
            res = res/np.sum(res + 10**(-10))
            res = (np.log(float(1)/(1-res)))
        
        final_res = []
        for not_kill in np.asarray(res):
            final_res.append(float(not_kill))
        return final_res

schema = StructType([
    StructField("F1", FloatType(), False),
    StructField("F2", FloatType(), False),
    StructField("F3", FloatType(), False),
    StructField("F4", FloatType(), False),
    StructField("F5", FloatType(), False),
    StructField("F6", FloatType(), False),
    StructField("F7", FloatType(), False),
    StructField("F8", FloatType(), False),
    StructField("F9", FloatType(), False)
])

toDWT_relative = partial(toDWT, rel = True)
toDWT_cont = partial(toDWT, rel = False)

toDWT_relative_udf = udf(toDWT_relative,  schema)
toDWT_cont_udf = udf(toDWT_cont,  schema)

def to_array(col):
    def to_array_(v):
        return v
    return udf(to_array_, ArrayType(FloatType()))(col)