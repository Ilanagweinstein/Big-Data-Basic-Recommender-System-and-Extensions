#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Evaluate MAP for ALS model in 2 ways 
    
    Parameters
    ----------
    train_data: parquet file that turns into spark DF with columns ['user_id', 'book_id', 'rating']
    
    validation_data: parquet file that turns into spark DF with columns ['user_id', 'book_id', 'rating']
    
    regParams: list of floats 
    
    ranks: list of ints 
    
    Return
    ------
    Print all the results. 
    Example: spark-submit evaluations.py hdfs:/user/smt570/small_train.parquet hdfs:/user/smt570/small_val.parquet 0.1,1 2,25
    
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

def main(spark, train_data, validation_data, regParams, ranks):
    spark_session = SparkSession.builder.appName('train').master('yarn').config('spark.executor.memory', '10g').config('spark.driver.memory', '10g').getOrCreate()
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
    
    for rank in ranks:
        for reg in regParams:
            # Build ALS model
            print('Building ALS model...')
            als=ALS(maxIter=5,regParam=reg,rank=rank,userCol="user_id",itemCol="book_id",ratingCol="rating",coldStartStrategy="drop",nonnegative=True)
            
            #Train ALS model
            print('Training ALS model...')
            model = als.fit(train_data)

            # Make predictions on val_data
            print('Making predictions...')
            predictions = model.transform(val_data)

            # Evaluate with RSME and print the results
            print('Evaluating...')      
            evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
            rmse = evaluator.evaluate(predictions)
            
            print('{} latent factors and regularization = {}: '
                              'validation RMSE is {}'.format(rank, reg, rmse))
                
            ###
            #MAP (Method 2)
        
            #model makes arbitary number of predictions
            preds = model.transform(val_data)
            
            #build predictions df: use rank to limit the number of predictions to top 500 for each user
            w = Window.partitionBy('user_id').orderBy(col('prediction').desc())
            preds = preds.select('user_id','book_id','prediction',F.rank().over(w).alias('rank')).where('rank <=500')
            
            #build predictions df: group books by user_id, store as single array of books in rating column
            predicted = preds.withColumn("sorted_vals_by_rating", F.collect_list('book_id').over(w))
            predicted = predicted.groupBy('user_id').agg(F.max('sorted_vals_by_rating').alias('items'))
            
            #build df of predictions and ground truth, convert to RDD
            joined_users = predicted.join(actual, 'user_id').select('user_id',predicted.items.alias('pred'),actual.items.alias('actual'))
            metrics = RankingMetrics(joined_users.select('pred','actual').rdd.map(tuple))
            m_ap2 = metrics.meanAveragePrecision
            
            print('{} latent factors and regularization = {}: '
                              'validation MAP is {}'.format(rank, reg, m_ap2))
            
            ####
            #MAP (Method 1)
            predictions = model.transform(val_data)
            
            #model makes top 500 predictions for all users
            preds = model.recommendForAllUsers(500)
            
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
            m_ap1 = rankingMetrics.meanAveragePrecision 
            
            print('{} latent factors and regularization = {}: '
                              'validation MAP is {}'.format(rank, reg, m_ap1))
    

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
