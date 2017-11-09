# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 08:52:14 2017

@author: imazeh
"""

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("some_testing2").master("local").getOrCreate()


df = spark.read.format('com.databricks.spark.csv').option("header", "True").option("delimiter", ",")\
                      .load('C:/Users/imazeh/Desktop/'+'clinic_assessments_watch_acc_data_old.csv') 
