# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 09:51:40 2017

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
from spark import utils_function_spark
from spark import proj_for_spark

def casting_data(df):
    """
    Casting of the data to double timestamp, unix or long
    """
    df = df.withColumn("X", df["X"].cast("double"))
    df = df.withColumn("Y", df["Y"].cast("double"))
    df = df.withColumn("Z", df["Z"].cast("double"))
    df = df.withColumn("TremorGA", df["TremorGA"].cast("double"))
    df = df.withColumn("BradykinesiaGA", df["BradykinesiaGA"].cast("double"))
    df = df.withColumn("DyskinesiaGA", df["DyskinesiaGA"].cast("double"))
    df = df.withColumn("TSStart", df["TSStart"].cast("timestamp"))
    df = df.withColumn("TSEnd", df["TSEnd"].cast("timestamp"))
    df = df.withColumn("interval_start", ((ceil(unix_timestamp(df["TSStart"]).cast("long")))%10**8)) 
    df = df.withColumn("interval_end", ((ceil(unix_timestamp(df["TSEnd"]).cast("long")))%10**8)) 
    df = df.withColumn("temp", utils_function_spark.find_milisec_udf('TS')) 
    df = df.withColumn("interval", (((unix_timestamp(df["TS"]).cast("long"))))) 
    df = df.withColumn("interval", utils_function_spark.merge_integers_udf('interval', 'temp'))
    df = df.withColumn("key", utils_function_spark.give_my_key_udf("interval_start", "interval_end", 'SubjectId') ) 
    df = df.withColumn("key", df["key"].cast("double"))
    
    return df


def make_inteval_per_task(df):
    """
    Aggragate the samples for intervals
    """
    
    ##create rdd with key = (key, tremor, bradykinesia, dyskinesia)
    # Values = (X, Y, Z, interval)
    df = df.select('key','TremorGA', 'BradykinesiaGA', 'DyskinesiaGA',
                   'X', 'Y', 'Z', 'interval').rdd.map\
                 (lambda raw: ((raw[0], raw[1], raw[2], raw[3]),
                               ([raw[4]], [raw[5]],  [raw[6]],  [raw[7]])))
                 
        
    ##Aggraate the data to intervals here, still RDD    
    df = df.reduceByKey(lambda x, y:\
                        (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3]))


    ##make the data as DF with the key Tremor Bradykinesia, Dyskinesia 
    ## And the new intervals X, Y, Z, interval
    df = df.map(lambda row : (row[0][0], row[0][1], row[0][2], row[0][3] ,row[1][0], row[1][1], row[1][2], row[1][3])).\
                       toDF(['key', 'TremorGA', 'BradykinesiaGA', 'DyskinesiaGA','X', 'Y', 'Z', 'interval'])
                   
    
    ##Sort each interval with respect to interval                
    df = df.withColumn('X', utils_function_spark.sort_vec_udf('X', 'interval'))
    df = df.withColumn('Y', utils_function_spark.sort_vec_udf('Y', 'interval'))
    df = df.withColumn('Z', utils_function_spark.sort_vec_udf('Z', 'interval')) 
    
    ##make String to array for each column
    df = df.withColumn("X",  utils_function_spark.parse_udf("X"))
    df = df.withColumn("Y",  utils_function_spark.parse_udf("Y"))
    df = df.withColumn("Z",  utils_function_spark.parse_udf("Z"))                  
    
    return df                   

def tasks_to_intervals(df, slide, window):
    """
    For each intervals create sliding window
    """
    sliding_window_udf = utils_function_spark.sliding_window(slide = slide, window_size = window)
    df = df.withColumn('X', sliding_window_udf('X', 'interval'))
    df = df.withColumn('Y', sliding_window_udf('Y', 'interval'))
    df = df.withColumn('Z', sliding_window_udf('Z', 'interval'))

    
    ##arrange X, Y, Z, togother and then flat the data to data new rdd with 
    ##the same key for each sullist 
    df = df.rdd.map(lambda raw:  ((raw[0], raw[1], raw[2], raw[3]) , 
                                         list(zip(raw[4], raw[5], raw[6])))).\
                          flatMapValues(lambda raw :raw)

    df = df.map(lambda raw: (raw[0],raw[1][0],raw[1][1],raw[1][2])).\
               toDF(['key', 'X', 'Y', 'Z'])
               
    return df

def get_features(df):
    """
    Proj Denoise and feature extraction on X, Y and Z from the data frame we have
    after tasks_to_intervals
    
    """
    
    schema = StructType([
    StructField("proj_ver", VectorUDT(), False),
    StructField("proj_hor", VectorUDT(), False)
    ])

    proj_func = udf(proj_for_spark.project_gravity_xyz, schema)

    df = df['X', 'Y', 'Z', 'key'].withColumn('proj', proj_func("X", "Y", "Z"))
    df = df.select('key',
                 'proj.proj_ver', 
                 'proj.proj_hor')
    
    df = df['proj_ver','proj_hor', 'key'].withColumn('denoised_ver',
                    utils_function_spark.denoise_func("proj_ver")).withColumn('denoised_hor', 
                                utils_function_spark.denoise_func("proj_hor"))
    df = df.select('key', "denoised_ver", "denoised_hor") 
    
    df = df["denoised_ver", "denoised_hor", 'key'].withColumn('rel_features_ver', 
                        utils_function_spark.toDWT_relative_udf("denoised_ver")).\
                        withColumn('cont_features_ver',
                        utils_function_spark.toDWT_cont_udf("denoised_ver"))

    df = df["rel_features_ver", "cont_features_ver", "denoised_hor", 'key'].\
                       withColumn('rel_features_hor', 
                       utils_function_spark.toDWT_relative_udf("denoised_hor")).\
                       withColumn('cont_features_hor',
                       utils_function_spark.toDWT_cont_udf("denoised_hor"))


    df = df.select('key', 'rel_features_ver', 'cont_features_ver',
                                 'rel_features_hor', 'cont_features_hor')

    
    return df
    
"""
def ready_for_model(df):
    return df\
    .withColumn("rel_features_ver", utils_function_spark.to_array(col("rel_features_ver")))\
    .withColumn("cont_features_ver", utils_function_spark.to_array(col("cont_features_ver")))\
    .withColumn("rel_features_hor", utils_function_spark.to_array(col("rel_features_hor")))\
    .withColumn("cont_features_hor", utils_function_spark.to_array(col("cont_features_hor")))\
    .select(["key"] + [col("rel_features_ver")[i] for i in range(9)] +
            [col("cont_features_ver")[i] for i in range(9)] + 
            [col("rel_features_hor")[i] for i in range(9)] +
            [col("cont_features_hor")[i] for i in range(9)])   

    
def prerpre_meta(df):
    return df.select('key','TremorGA', 'BradykinesiaGA', 'DyskinesiaGA').\
                     rdd.map(lambda raw: (raw[0], (raw[1], raw[2], raw[3]))).\
                     reduceByKey(lambda x, y: (x[0], x[1], x[2])).\
                     map(lambda raw: (raw[0],raw[1][0], raw[1][1], raw[1][2])).\
                     toDF(['key', 'TremorGA', 'BradykinesiaGA', 'DyskinesiaGA'])
"""
