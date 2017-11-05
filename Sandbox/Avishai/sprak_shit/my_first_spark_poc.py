# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 10:49:37 2017

@author: awagner
"""
import numpy as np
import pyspark
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

spark = SparkSession.builder.appName("some_testing").master("local").getOrCreate()

df = spark.read.format('com.databricks.spark.csv').option("header", "True").option("delimiter", ",")\
                      .load('C:/Users/awagner/Desktop/For_Tom/'+'AllLabData.csv')

df = df.withColumn("X", df["X"].cast("double"))
df = df.withColumn("Y", df["Y"].cast("double"))
df = df.withColumn("Z", df["Z"].cast("double"))
df = df.withColumn("TS", df["TS"].cast("timestamp"))

df = df.withColumn("date", df["TS"].cast("timestamp")).withColumn("interval", ((ceil(unix_timestamp(df["TS"]).cast("long") / 5))*5.0).cast("timestamp")) 

window = Window.partitionBy(df['date']).orderBy(df['TS'])

def unnest(col1):
  
  res =[]
  for i in range(len(col1)):
      res.append(col1[i][1])

  return(res)

unnest_udf = udf(unnest)

df2 = df.select('*', rank().over(window).alias('rank')) \
  .groupBy("interval") \
  .agg(collect_list(array("rank","X")).alias("X"), collect_list(array("rank", "Y")).alias("Y"), 
      collect_list(array("rank", "Z")).alias("Z")) \
  .withColumn("X", unnest_udf("X")) \
  .withColumn("Y", unnest_udf("Y")) \
  .withColumn("Z", unnest_udf("Z")) \
  .sort("interval") 
  
parse = udf(lambda s: Vectors.dense([float(c) for c in s.replace("[","").replace("]","").split(",")]), VectorUDT())
parse2 = udf(lambda s: Vectors.dense(eval(str(s))), VectorUDT())

df2 = df2.withColumn("X", parse2("X"))
df2 = df2.withColumn("Y", parse2("Y"))
df2 = df2.withColumn("Z", parse2("Z"))

df2.cache()

def squared(x, y, z):
    norm = []
    for i in range(len(x)):
        norm.append(float(pow(x[i],2) + pow(y[i],2) + pow(z[i],2)))
    
    return norm

squared_udf = udf(squared)
df3 = df2['X', 'Y', 'Z', 'interval'].withColumn('norma', squared_udf("X","Y","Z"))
df3 = df3.filter(df3.norma != df3.take(1)[0]['norma'])

########################################################################################
schema = StructType([
    StructField("proj_ver", VectorUDT(), False),
    StructField("proj_hor", VectorUDT(), False)
])

#proj_new = partial(project_gravity_core, rel = True)
proj_func = udf(project_gravity_xyz, schema)

df_proj = df3['X', 'Y', 'Z', 'interval'] .withColumn('proj', proj_func("X", "Y", "Z"))
df_proj = df_proj.select('interval',
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

denoise_func = udf(denoise, VectorUDT())



df_denoise = df_proj['proj_ver','proj_hor', 'interval'].withColumn('denoised_ver',
                    denoise_func("proj_ver")).withColumn('denoised_hor',denoise_func("proj_hor"))
df_denoise = df_denoise.select('interval', "denoised_ver", "denoised_hor") 
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

df_features = df_denoise["denoised_ver", "denoised_hor", 'interval'].withColumn('rel_features_ver', 
                        toDWT_relative_udf("denoised_ver")).withColumn('cont_features_ver',
                                          toDWT_cont_udf("denoised_ver"))

df_features = df_features["rel_features_ver", "cont_features_ver", "denoised_hor", 'interval'].withColumn('rel_features_hor', 
                        toDWT_relative_udf("denoised_hor")).withColumn('cont_features_hor',
                                          toDWT_cont_udf("denoised_hor"))


df_features = df_features.select('interval', 'rel_features_ver', 'cont_features_ver',
                                 'rel_features_hor', 'cont_features_hor')
df_features.show(2)

####################################################################################

from pyspark.sql.window import Window
from pyspark.sql.functions import percent_rank
df4=df3.select('norma', 'interval', percent_rank().over(Window.orderBy(df3.interval)).\
              alias("interval_perc"),
              percent_rank().over(Window.orderBy(df.DEBIT)).alias("debit_perc"))\
  .where('debit_perc >=0.99 or debit_perc <=0.01 ').where\
        ('credit_perc >=0.99 or credit_perc <=0.91')

####################################################################################

