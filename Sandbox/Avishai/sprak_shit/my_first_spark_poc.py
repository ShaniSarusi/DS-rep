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

df2 = df2.withColumn("X", parse("X"))
df2 = df2.withColumn("Y", parse("Y"))
df2 = df2.withColumn("Z", parse("Z"))

df2.cache()

def squared(x, y, z):
    norm = []
    for i in range(len(x)):
        norm.append(float(pow(x[i],2) + pow(y[i],2) + pow(z[i],2)))
    
    return norm

squared_udf = udf(squared)
df3 = df2['X', 'Y', 'Z', 'interval'].withColumn('norma', squared_udf("X","Y","Z"))


########################################################################################
from scipy.interpolate import interpolate
import pywt
from future.utils import lmap
import numpy as np
from functools import partial

#some_test = df3.toPandas()
df4 = df3.filter(df3.norma != some_test['norma'][0])
df4 = df4.withColumn("norma", parse("norma"))

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

toDWT_new = partial(toDWT, rel = True)
toDWT_udf = udf(toDWT_new,  schema)

df4 = df4['norma', 'interval'].withColumn('wav_features', toDWT_udf("norma"))
df4 = df4.select('interval',
                 'wav_features.F1', 
                 'wav_features.F2',
                 'wav_features.F3',
                 'wav_features.F4',
                 'wav_features.F5',
                 'wav_features.F6',
                 'wav_features.F7',
                 'wav_features.F8',
                 'wav_features.F9')
df4.show(2)

####################################################################################

from pyspark.sql.window import Window
from pyspark.sql.functions import percent_rank
df4=df3.select('norma', 'interval', percent_rank().over(Window.orderBy(df3.interval)).\
              alias("interval_perc"),
              percent_rank().over(Window.orderBy(df.DEBIT)).alias("debit_perc"))\
  .where('debit_perc >=0.99 or debit_perc <=0.01 ').where\
        ('credit_perc >=0.99 or credit_perc <=0.91')
