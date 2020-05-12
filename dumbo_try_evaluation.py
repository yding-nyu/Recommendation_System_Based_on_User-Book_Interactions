### import packages

import os
import time

# spark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import UserDefinedFunction, explode, desc
from pyspark.sql.types import StringType, ArrayType
# from pyspark.mllib.recommendation import ALS
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.sql.functions import col

from pyspark import SparkConf, Row

from pyspark.sql.functions import desc
from pyspark.sql.functions import col
from pyspark.sql.functions import *
import glob
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.context import SparkContext
from pyspark.serializers import MarshalSerializer
from pyspark.mllib.evaluation import RankingMetrics

# data science imports
import math
import numpy as np
import pandas as pd

#spark session
spark = SparkSession \
    .builder \
    .appName("movie recommendation") \
    .config("spark.driver.maxResultSize", "96g") \
    .config("spark.driver.memory", "96g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.master", "local[12]") \
    .getOrCreate()
# get spark context
sc = spark.sparkContext


valid = spark.read.csv('valid.csv', header = True)
train = spark.read.csv('train.csv', header = True)
test = spark.read.csv('test.csv', header = True)

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
#transformed_train.show() 

transformed_train.createOrReplaceTempView('transformed_train_view')
transformed_valid.createOrReplaceTempView('transformed_valid_view')
transformed_test.createOrReplaceTempView('transformed_test_view')

Train = transformed_train.withColumn("rating", transformed_train.rating.cast("int"))
Valid = transformed_valid.withColumn("rating", transformed_valid.rating.cast("int"))
Test = transformed_test.withColumn("rating", transformed_test.rating.cast("int"))


ALSExplicit = ALS(userCol="user_id_index",itemCol="book_id_index",ratingCol="rating",coldStartStrategy="drop",nonnegative=True)
defaultModel = ALSExplicit.fit(Train)

paramMapExplicit = ParamGridBuilder() \
                    .addGrid(ALSExplicit.rank, [1,5,8,10,15,20,25,30,35,40,45,50,100,150]) \
                    .addGrid(ALSExplicit.regParam, [.01, .05, .1, .15, .2, .25]) \
                    .build()
evaluatorR = RegressionEvaluator(metricName="rmse", labelCol="rating")

# Run cross-validation, and choose the best set of parameters.
CVALSExplicit = CrossValidator(estimator=ALSExplicit,
                            estimatorParamMaps=paramMapExplicit,
                            evaluator=evaluatorR,
                           numFolds=8)

CVModelEXplicit = CVALSExplicit.fit(Train)
predsExplicit = CVModelEXplicit.bestModel.transform(Test)


top500 = CVModelEXplicit.bestModel.recommendForAllUsers(500)
top500.createOrReplaceTempView('top500')
actual = Train.groupBy("user_id_index").agg(expr("collect_set(book_id_index) as book"))
pred_500 = top500.select('user_id_index', "recommendations.book_id_index")
joining_500 = pred_500.join(actual, ['user_id_index']).select('book_id_index','book')
metrics = RankingMetrics(joining_500.rdd)
precision_at_500 = metrics.precisionAt(500)
MAP = metrics.meanAveragePrecision
print("MAP: {}, Precision-at-500: {}".format(MAP, precision_at_500))
