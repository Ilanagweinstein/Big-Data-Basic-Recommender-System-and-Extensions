#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Comparing LightFM Model to ALS Model
    
    Parameters
    ----------
    train_data: parquet file that turns into spark DF with columns ['user_id', 'book_id', 'rating']
    
    validation_data: parquet file that turns into spark DF with columns ['user_id', 'book_id', 'rating']
    
    
    Return
    ------
    Print model fitting time and precision@k. 
    Example: spark-submit extension1.py hdfs:/user/smt570/small_train.parquet hdfs:/user/smt570/small_val.parquet
    
'''

import sys
import argparse

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark import SparkContext
from sklearn import preprocessing
import numpy as np
import pandas as pd
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import udf
from scipy.sparse import coo_matrix 
from lightfm import LightFM
from pyspark.ml.recommendation import ALS
import time
from pyspark.sql.functions import col
import pyspark.sql.functions as F
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Window
from pyspark.sql.functions import col, expr, explode
import pyspark.sql.functions as F
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics


def main(spark, train_data, validation_data):
    spark_session = SparkSession.builder.appName('extension1').master('yarn').config('spark.executor.memory', '15g').config('spark.driver.memory', '15g').getOrCreate()
    spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    
    #####################################################################
    #LIGHTFM Model
    
    # Read data from parquet
    train_df = spark.read.parquet('hdfs:/user/smt570/small_train.parquet')
    train_df.createOrReplaceTempView('train')
    train_df = train_df.select('user_id','book_id','rating')

    val = spark.read.parquet('hdfs:/user/smt570/small_val.parquet')
    val.createOrReplaceTempView('val')
    val_df = val.select('user_id','book_id','rating')
    #remove ratings less than 3 from ground truth
    val_df = val_df.filter(val_df.rating >= 3)

    #all positive instances for training (rating >=3) keep their values, anything else becomes 0
    eq = udf(lambda x: x if x >=3 else 0, IntegerType())
    train_df = train_df.withColumn('rating',eq(train_df.rating))

    #need to sort first
    train_df = train_df.orderBy('user_id')
    
    print('Building input sparse matrices...')
    #convert to pandas for pre-processing
    train_df = train_df.toPandas()
    val_df = val_df.toPandas()

    #initialize dicts
    transf_train = dict()
    transf_val = dict()

    enc = preprocessing.LabelEncoder()

    #transform data values for train and val
    transf_train['user_id']=enc.fit_transform(train_df['user_id'].values)
    transf_train['book_id'] = enc.fit_transform(train_df['book_id'].values)
    transf_train['rating']=enc.fit_transform(train_df['rating'].values)

    transf_val['user_id']=enc.fit_transform(val_df['user_id'].values)
    transf_val['book_id'] = enc.fit_transform(val_df['book_id'].values)
    transf_val['rating']=enc.fit_transform(val_df['rating'].values)

    #get size of COO matrix
    n_users = len(np.unique(transf_train['user_id']))
    n_items = len(np.unique(transf_train['book_id']))

    #create COO matrices 
    train = coo_matrix((transf_train['rating'],(transf_train['user_id'],transf_train['book_id'])),shape=(n_users,n_items))
    val = coo_matrix((transf_val['rating'],(transf_val['user_id'],transf_val['book_id'])),shape=(n_users,n_items))

    #Build LightFM model
    print('Building LightFM model...')
    model = LightFM(loss = 'warp', no_components = 30)

    #Train LightFM model and check time to fit
    print('Training LightFM model...')
    start_time = time.time()
    model.fit(train)

    print('Run time: {} mins'.format((time.time() - start_time)/60))

    #Get data ready for evaluation, use top k predictions for metrics
    print('Evaluating...')
    pak_train = precision_at_k(model,train,k=125).mean()
    pak_val = precision_at_k(model,val,k=125).mean()

    print('Train precision@K = {}:'.format(pak_train))
    print('Test precision@K = {}:'.format(pak_val))
    
    auc_train = auc_score(model, train).mean()
    auc_test = auc_score(model, val).mean()

    print("Train AUC Score: {}".format(auc_train))
    print("Test AUC Score: {}".format(auc_test))
    
    ###################################################################
    #ALS Model
    
    # Read data from parquet
    train = spark.read.parquet(train_data)
    train.createOrReplaceTempView('train')
    train_data = train.select('user_id','book_id','rating')
    train_data = train_data.filter(train_data.rating !=0)

    val = spark.read.parquet(validation_data)
    val.createOrReplaceTempView('val')
    val_data = val.filter(val.rating >= 3)
    val_data = val.select('user_id','book_id','rating')
    
    #creating ground truth df
    w = Window.partitionBy('user_id').orderBy(col('rating').desc())
    actual = val_data.withColumn("sorted_vals_by_rating", F.collect_list('book_id').over(w))
    actual = actual.groupBy('user_id').agg(F.max('sorted_vals_by_rating').alias('items'))
    
    # Go through parameters
    
    # Build ALS model
    print('Building ALS model...')
    als=ALS(maxIter=5,regParam=0.1,rank=2,userCol="user_id",itemCol="book_id",ratingCol="rating",coldStartStrategy="drop",nonnegative=True)

    #Train ALS model
    print('Training ALS model...')
    start_time = time.time()
    model = als.fit(train_data)

    print('Run time: {} mins'.format((time.time() - start_time)/60))

    # Make predictions on val_data
    print('Making predictions...')
    predictions = model.transform(val_data)

    ####
    #MAP (Method 1)
    predictions = model.transform(val_data)

    #model makes top k predictions for all users
    preds = model.recommendForAllUsers(125)

    #remove StructType
    preds = preds.withColumn('recommendations',explode('recommendations')).select('*')
    preds = preds.select('user_id','recommendations.*')

    #build predictions df: group books by user_id, store as single array of books in rating column
    w = Window.partitionBy('user_id').orderBy(col('rating').desc())
    perUserPredictedItemsDF = preds.select('user_id', 'book_id', 'rating', F.rank().over(w).alias('rank')).where('rank <= 500').groupBy('user_id').agg(expr('collect_list(book_id) as books'))
    windowSpec = Window.partitionBy('user_id').orderBy(col('rating').desc())
    perUserActualItemsDF = val.select('user_id', 'book_id', 'rating', F.rank().over(windowSpec).alias('rank')).groupBy('user_id').agg(expr('collect_list(book_id) as books')) 

    #build df of predictions and ground truth, convert to RDD
    perUserItemsRDD = perUserPredictedItemsDF.join(perUserActualItemsDF, 'user_id').rdd.map(lambda row: (row[1], row[2]))
    rankingMetrics = RankingMetrics(perUserItemsRDD)                            
    pak = rankingMetrics.precisionAt(125)

    print('Precision at k is {}'.format(pak))



# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('extension1').getOrCreate()
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('train_data')
    parser.add_argument('validation_data')
    
    args = parser.parse_args()
    
    train_data = args.train_data
    validation_data = args.validation_data

    # Call our main routine
    main(spark, train_data, validation_data)
