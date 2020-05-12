#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import packages
import os
import time
import glob
import math
import numpy as np
import pandas as pd
import pickle as pkl
import itertools as it

# spark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import UserDefinedFunction, explode, desc
from pyspark.sql.types import StringType, ArrayType
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark import SparkConf, Row
from pyspark.sql.functions import desc
from pyspark.sql.functions import col
import pyspark.sql.functions as F
from pyspark.sql.functions import expr
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.context import SparkContext
from pyspark.serializers import MarshalSerializer
from pyspark.mllib.evaluation import RankingMetrics


#spark session
spark = SparkSession \
    .builder \
    .appName("movie recommendation") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.memory.offHeap.enabled","true")\
    .config("spark.memory.offHeap.size","4g")\
    .config("spark.driver.memoryOverhead", "4g")\
    .config("spark.executor.memoryOverhead", "4g")\
    .config("spark.dynamicAllocation.enabled", "false")\
    .getOrCreate()
# get spark context
sc = spark.sparkContext

## Load in datasets
train_path = 'hdfs:/user/ss13289/train1.parquet'
val_path = 'hdfs:/user/ss13289/valid1.parquet'
test_path = 'hdfs:/user/ss13289/test1.parquet'

train = spark.read.parquet(train_path)
valid = spark.read.parquet(val_path)
test = spark.read.parquet(test_path)


train_ = train.select(train['user_id'], train['book_id'], train['rating'])
valid_ = valid.select(valid['user_id'], valid['book_id'], valid['rating'])
test_ = test.select(test['user_id'], test['book_id'], test['rating'])


indexer_va = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in list(set(valid_.columns)-set(['rating']))]
indexer_tr = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in list(set(train_.columns)-set(['rating']))]
indexer_te = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in list(set(test_.columns)-set(['rating']))]

# Converting String to index
pipeline = Pipeline(stages=indexer_tr)
transformed_train= pipeline.fit(train_).transform(train_)

pipeline = Pipeline(stages=indexer_va)
transformed_valid= pipeline.fit(valid_).transform(valid_)

pipeline = Pipeline(stages=indexer_te)
transformed_test= pipeline.fit(test_).transform(test_)


transformed_train.createOrReplaceTempView('transformed_train_view')
transformed_valid.createOrReplaceTempView('transformed_valid_view')
transformed_test.createOrReplaceTempView('transformed_test_view')

Train = transformed_train.withColumn("rating", transformed_train.rating.cast("int"))
Valid = transformed_valid.withColumn("rating", transformed_valid.rating.cast("int"))
Test = transformed_test.withColumn("rating", transformed_test.rating.cast("int"))

# ALS parameters
rank_  = [5, 10, 20]
regParam_ = [0.01, 0.1, 1]
alpha = [1,5,10]
param_grid = it.product(rank_, regParam_, alpha)

# validation set
user_id = Valid.select('user_id_index').distinct()
true_label = Valid.groupBy('user_id_index').agg(expr("collect_set(book_id_index) as book"))


for i in param_grid:
    print('Start Training for {}'.format(i))
    als = ALS(rank = i[0], maxIter=10, regParam=i[1], alpha = i[2], userCol="user_id_index", itemCol="book_id_index", ratingCol="rating", implicitPrefs=True, nonnegative=True, coldStartStrategy="drop")
    model = als.fit(Train)
    print('Finish Training for {}'.format(i))
    res = model.recommendForUserSubset(user_id,500)
    pred_label = res.select('user_id_index', "recommendations.book_id_index")
    pred_true = pred_label.join(true_label, ['user_id_index']).select('book_id_index','book')
    print("-"*30)
    print('Start Evaluataion for {}'.format(i))
    metrics = RankingMetrics(pred_true.rdd)
    MAP = metrics.meanAveragePrecision
    precision_at_500 = metrics.precisionAt(500)
    print("MAP: {}, Precision-at-500: {}".format(MAP, precision_at_500))

print("-"*30)
print("Tuning finished !")
print("-"*30)


# After tuning, choose the best model and get the performance on test set
als = ALS(rank = 20, maxIter=10, regParam= 1, alpha = 1, userCol="user_id_index", itemCol="book_id_index", ratingCol="rating", implicitPrefs=True, nonnegative=True, coldStartStrategy="drop")
user_id = Test.select('user_id_index').distinct()
true_label = Test.groupBy('user_id_index').agg(expr("collect_set(book_id_index) as book"))

print("Start Training for the Best Model")
res = model.recommendForUserSubset(user_id,500)
pred_label = res.select('user_id_index', "recommendations.book_id_index")
pred_true = pred_label.join(true_label, ['user_id_index']).select('book_id_index','book')
print("-"*30)
print("Evaluation for Test Set")
metrics = RankingMetrics(pred_true.rdd)
MAP = metrics.meanAveragePrecision
precision_at_500 = metrics.precisionAt(500)
print("MAP: {}, Precision-at-500: {}".format(MAP, precision_at_500))


# retrieve latent factors from the model
user_factor = model.userFactors
user_all = user_factor.toPandas()
user_vec = np.array(list(user_all['features']))
pkl.dump(user_vec, open('user_vec.pkl','wb'))
pkl.dump(user_all, open('user_all.pkl','wb'))


item_factor = model.itemFactors
item_all = item_factor.toPandas()
item_vec = np.array(list(item_all['features']))
pkl.dump(item_vec, open('item_vec.pkl','wb'))
pkl.dump(item_all, open('item_all.pkl','wb'))
print("-"*30)
print("All finished!")
print("-"*30)