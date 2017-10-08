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
        norm.append(pow(x[i],2) + pow(y[i],2) + pow(z[i],2))
    
    return np.mean(norm)

squared_udf = udf(squared, FloatType())
df3 = df2['X', 'Y', 'Z', 'interval'].withColumn('norma', squared_udf("X","Y","Z"))
df3 = df2.select(squared_udf("X","Y","Z"))

