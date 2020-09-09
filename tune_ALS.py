#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Hyperparameter tuning Recommendation system

    grid search function to select the best model based on RMSE of
    validation data
    
    Parameters
    ----------
    train_data: parquet file that turns into spark DF with columns ['userId', 'movieId', 'rating']
    
    validation_data: parquet file that turns into spark DF with columns ['userId', 'movieId', 'rating']
    
    regParams: list of floats 
    
    ranks: list of ints 
    
    Return
    ------
    Print all the results. 

    Example: spark-submit tune_ALS.py hdfs:/user/smt570/small_train.parquet hdfs:/user/smt570/small_val.parquet 0.1,1 2,25
    
'''

import sys
import argparse

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Window
from pyspark.sql.functions import col, expr, explode
import pyspark.sql.functions as F
from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
import time

def main(spark, train_data, validation_data, regParams, ranks):
    spark_session = SparkSession.builder.appName('train').master('yarn').config('spark.executor.memory', '10g').config('spark.driver.memory', '10g').getOrCreate()
    # Read data from parquet
    train = spark.read.parquet(train_data)
    train.createOrReplaceTempView('train')
    train_data = train.select('user_id','book_id','rating')
    train_data = train_data.filter(train_data.rating !=0)

    val = spark.read.parquet(validation_data)
    val.createOrReplaceTempView('val')
    val_data = val.select('user_id','book_id','rating')
    val_data = val_data.filter(val_data.rating >= 3)
    
    #create window to order books based on how confident we are in predictions; MAP cares about order
    w = Window.partitionBy('user_id').orderBy(col('rating').desc())
    
    #create actual df from val_data: group books by user_id, store as single array in rating column
    actual = val_data.withColumn("sorted_vals_by_rating", F.collect_list('book_id').over(w))
    actual = actual.groupBy('user_id').agg(F.max('sorted_vals_by_rating').alias('items'))
    
    # Go through parameters
    rmse_list = []
    rank_list = []
    regParams_list = []
    
    for rank in ranks:
        for reg in regParams:
            startTimeQuery = time.clock()
            
            # Build ALS model
            print('Building ALS model...')
            als=ALS(maxIter=5,regParam=reg,rank=rank,userCol="user_id",itemCol="book_id",ratingCol="rating",coldStartStrategy="drop",nonnegative=True)
            
            rank_list.append(rank)
            regParams_list.append(reg)
            
            #Train ALS model
            print('Training ALS model...')
            model = als.fit(train_data)

            # Make predictions on val_data
            print('Making predictions...')
            predictions = model.transform(val_data)

            #Tune the model with RSME and print the results
            print('Evaluating...')      
            evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
            rmse = evaluator.evaluate(predictions)
            rmse_list.append(rmse)
            
            endTimeQuery = time.clock()
            runTimeQuery = endTimeQuery - startTimeQuery
            
            print('{} latent factors and regularization = {}: '
                              'validation RMSE is {}. Calculated in {} seconds.'.format(rank, reg, rmse, runTimeQuery))
                
    
    print('Finding best model...')
    min_idx = rmse_list.index(min(rmse_list))
    print('Best model is with {} latent factors and regularization = {}: '
                              'validation RMSE is {}'.format(rank_list[min_idx], regParams_list[min_idx], rmse_list[min_idx]))
    

# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('supervised_train').getOrCreate()
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('train_data')
    parser.add_argument('validation_data')
    parser.add_argument('regParams')
    parser.add_argument('ranks')
    
    args = parser.parse_args()
    
    train_data = args.train_data
    validation_data = args.validation_data
    regParams = [float(item) for item in args.regParams.split(',')]
    ranks = [int(item) for item in args.ranks.split(',')]

    # Call our main routine
    main(spark, train_data, validation_data, regParams, ranks)
