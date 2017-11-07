# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:09:54 2017

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

spark = SparkSession.builder.appName("some_testing2").master("local").getOrCreate()

df = spark.read.format('com.databricks.spark.csv').option("header", "True").option("delimiter", ",")\
                      .load('C:/Users/awagner/Desktop/For_Tom/'+'AllLabData.csv')

df = df.withColumn("X", df["X"].cast("double"))
df = df.withColumn("Y", df["Y"].cast("double"))
df = df.withColumn("Z", df["Z"].cast("double"))
df = df.withColumn("TremorGA", df["TremorGA"].cast("double"))
df = df.withColumn("BradykinesiaGA", df["BradykinesiaGA"].cast("double"))
df = df.withColumn("DyskinesiaGA", df["DyskinesiaGA"].cast("double"))
df = df.withColumn("TSStart", df["TSStart"].cast("timestamp"))
df = df.withColumn("TSEnd", df["TSEnd"].cast("timestamp"))
df = df.withColumn("interval_start", ((ceil(unix_timestamp(df["TSStart"]).cast("long")))%)) 
df = df.withColumn("interval_end", ((ceil(unix_timestamp(df["TSEnd"]).cast("long"))))) 


schema = ArrayType(FloatType(), False)
parse2 = udf(lambda s: eval(str(s)), schema)
find_milisec = udf(lambda raw: (raw)[(raw.find('.')+1):(raw.find('.')+3)])
merge_integers = udf(lambda raw1, raw2: int(str(raw1) + str(raw2)))
df = df.withColumn("temp", find_milisec('TS')) 
df = df.withColumn("interval", (((unix_timestamp(df["TS"]).cast("long"))))) 
df = df.withColumn("interval", merge_integers('interval', 'temp'))



def give_my_key(*args):
    key = 0
    for i in args:
        key += float(i)
    return key

give_my_key_udf = udf(give_my_key)
    

df = df.withColumn("key", give_my_key_udf("interval_start", "interval_end", 'SubjectId') ) 
df = df.withColumn("key", df["key"].cast("double"))

df.cache()
rdd_test = df.select('key','TremorGA', 'BradykinesiaGA', 'DyskinesiaGA','X', 'Y', 'Z', 'interval').rdd.map\
                 (lambda raw: ((raw[0], raw[1], raw[2], raw[3]),
                               ([raw[4]], [raw[5]],  [raw[6]],  [raw[7]])))
                 

rdd_test2 = rdd_test.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]))



df_test = rdd_test2.map(lambda row : (row[0][0], row[0][1], row[0][2], row[0][3] ,row[1][0], row[1][1], row[1][2], row[1][3])).\
                       toDF(['key', 'TremorGA', 'BradykinesiaGA', 'DyskinesiaGA','X', 'Y', 'Z', 'interval'])
                       
                       
sort_vec = udf(lambda X, Y: [x for _,x in sorted(zip(Y,X))])

df_test2 = df_test.withColumn('X', sort_vec('X', 'interval'))
df_test2 = df_test2.withColumn('Y', sort_vec('Y', 'interval'))
df_test2 = df_test2.withColumn('Z', sort_vec('Z', 'interval'))

df_test2 = df_test2.withColumn("X",  parse2("X"))
df_test2 = df_test2.withColumn("Y",  parse2("Y"))
df_test2 = df_test2.withColumn("Z",  parse2("Z"))

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

schema = ArrayType(ArrayType(FloatType(), False), False)

sliding_window_new = partial(slidind_window, slide = 2.5, window_size = 5)

sliding_window_udf = udf(sliding_window_new, schema)
               
df_test3 = df_test2.withColumn('X', sliding_window_udf('X', 'interval'))
df_test3 = df_test3.withColumn('Y', sliding_window_udf('Y', 'interval'))
df_test3 = df_test3.withColumn('Z', sliding_window_udf('Z', 'interval'))


df_flat = df_test3.rdd.map(lambda raw:  ((raw[0], raw[1], raw[2], raw[3]) , 
                                         list(zip(raw[4], raw[5], raw[6])))).\
                          flatMapValues(lambda raw :raw)

df_flat = df_flat.map(lambda raw: (raw[0],raw[1][0],raw[1][1],raw[1][2])).\
                     toDF(['key', 'X', 'Y', 'Z'])

   




########################################################################################
schema = StructType([
    StructField("proj_ver", VectorUDT(), False),
    StructField("proj_hor", VectorUDT(), False)
])

#proj_new = partial(project_gravity_core, rel = True)
proj_func = udf(project_gravity_xyz, schema)

df_proj = df_flat['X', 'Y', 'Z', 'key'].withColumn('proj', proj_func("X", "Y", "Z"))
df_proj = df_proj.select('key',
                 'proj.proj_ver', 
                 'proj.proj_hor')
df_proj.show(2)

########################################################################################
from scipy.signal import butter, filtfilt
from future.utils import lmap
import numpy as np
import pywt
import pandas as pd


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

schema = ArrayType(FloatType(), False)
denoise_func = udf(denoise,  VectorUDT())



df_denoise = df_proj['proj_ver','proj_hor', 'key'].withColumn('denoised_ver',
                    denoise_func("proj_ver")).withColumn('denoised_hor',denoise_func("proj_hor"))
df_denoise = df_denoise.select('key', "denoised_ver", "denoised_hor") 
df_denoise.show(2)


########################################################################################
from scipy.interpolate import interpolate
import pywt
from future.utils import lmap
import numpy as np
from functools import partial


#df4 = df3.withColumn("norma", parse2("norma"))

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

df_features = df_denoise["denoised_ver", "denoised_hor", 'key'].withColumn('rel_features_ver', 
                        toDWT_relative_udf("denoised_ver")).withColumn('cont_features_ver',
                                          toDWT_cont_udf("denoised_ver"))

df_features = df_features["rel_features_ver", "cont_features_ver", "denoised_hor", 'key'].withColumn('rel_features_hor', 
                        toDWT_relative_udf("denoised_hor")).withColumn('cont_features_hor',
                                          toDWT_cont_udf("denoised_hor"))


df_features = df_features.select('key', 'rel_features_ver', 'cont_features_ver',
                                 'rel_features_hor', 'cont_features_hor')
df_features.show(2)


#######################################################################################
def to_array(col):
    def to_array_(v):
        return v
    return udf(to_array_, ArrayType(FloatType()))(col)

ready_for_model = (df_features
    .withColumn("rel_features_ver", to_array(col("rel_features_ver")))
    .withColumn("cont_features_ver", to_array(col("cont_features_ver")))
    .withColumn("rel_features_hor", to_array(col("rel_features_hor")))
    .withColumn("cont_features_hor", to_array(col("cont_features_hor")))          
    .select(["key"] + [col("rel_features_ver")[i] for i in range(9)] + 
            [col("cont_features_ver")[i] for i in range(9)] + 
            [col("rel_features_hor")[i] for i in range(9)] +
            [col("cont_features_hor")[i] for i in range(9)]))


#########################################################################################
meta_data = df.select('key','TremorGA', 'BradykinesiaGA', 'DyskinesiaGA').\
                     rdd.map(lambda raw: (raw[0], (raw[1], raw[2], raw[3]))).\
                     reduceByKey(lambda x, y: (x[0], x[1], x[2])).\
                     map(lambda raw: (raw[0],raw[1][0], raw[1][1], raw[1][2])).\
                     toDF(['key', 'TremorGA', 'BradykinesiaGA', 'DyskinesiaGA'])
                     
    
joined_df = df_flat.join(meta_data, ['key'], 'inner')


####################3
ready_for_model = df_features.select('key.*', '*')
