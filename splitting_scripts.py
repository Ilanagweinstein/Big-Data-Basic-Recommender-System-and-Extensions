#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Data Splitting

    Splitting Professor McFee's data and then
    creating parque files for train, val, test
    
    Parameters
    ----------
    percent: percent as a float of original data to work with, so 1% of the data is .01
    
    Return
    ------
    Saves train, validation, trainvalCOMBO and test parquet files to hdfs 

    Example: spark-submit splitting_scripts.py .01
             spark-submit splitting_scripts.py .05
             spark-submit splitting_scripts.py .25
    
'''

import sys
import argparse

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F  

def main(spark, percent):

    # Read in data of users with is_reviewed > 0 (previously queried and parqueted)
    print('Reading in data...')
    books_large = spark.read.parquet('hdfs:/user/smt570/goodreads_interactions_ONLY_READ.parquet')
    books_large.createOrReplaceTempView('books_large')

    # Downsample by user percent
    print('Downsampling by user percent...')
    users = books_large.select('user_id').distinct()
    downsampled_users = users.sample(False, percent)
    data = downsampled_users.join(books_large, on=['user_id'], how='left')
    data = data.select('user_id', 'book_id', 'is_read', 'rating', 'is_reviewed')
    data.createOrReplaceTempView('data')

    # Remove users with less than 10 records
    print('Removing users with less than 10 records...')
    users10 = data.join(data.groupBy('user_id').count(),on='user_id')
    data = users10.filter(users10['count'] >= 10)
    data = data.select('user_id', 'book_id', 'is_read', 'rating', 'is_reviewed')

    # Data splitting
    print('Data splitting part 1...')
    users = data.select('user_id').distinct() 
    train_users, val_users, test_users = users.randomSplit([0.6, 0.2, 0.2], seed=2020)

    # Create first data sets
    train = train_users.join(data, on=['user_id'], how='left')
    val = val_users.join(data, on=['user_id'], how='left')
    test = test_users.join(data, on=['user_id'], how='left')

    # Splitting from val and test
    print('Data splitting part 2...')
    w1 = Window.partitionBy(val.user_id).orderBy(val.book_id)
    w2 = Window.partitionBy(val.user_id)

    w1 = Window.partitionBy(val.user_id).orderBy(val.book_id)
    w2 = Window.partitionBy(val.user_id)
    df = val.withColumn("row_number", F.row_number().over(w1)).withColumn("count", F.count('user_id').over(w2)).withColumn("percent", (F.col("row_number")/F.col("count")))
    real_val = df.filter(df.percent<=0.50)
    back2train1 = df.filter(df.percent>0.50)

    w1 = Window.partitionBy(test.user_id).orderBy(test.book_id)
    w2 = Window.partitionBy(test.user_id)
    df = test.withColumn("row_number", F.row_number().over(w1)).withColumn("count", F.count('user_id').over(w2)).withColumn("percent", (F.col("row_number")/F.col("count")))
    real_test = df.filter(df.percent<=0.50)
    back2train2 = df.filter(df.percent>0.50)

    # Remake train set
    back2train1 = back2train1.select('user_id', 'book_id', 'is_read', 'rating', 'is_reviewed')
    back2train2 = back2train2.select('user_id', 'book_id', 'is_read', 'rating', 'is_reviewed')
    train = train.union(back2train1)
    train = train.union(back2train2)

    # Write train, val and test to parquet
    print('Writing out to HDFS...')
    train = train.select('user_id', 'book_id', 'is_read', 'rating', 'is_reviewed')
    train.write.parquet('hdfs:/user/smt570/' + str(int(percent*100)) + '_train.parquet')
    
    real_val = real_val.select('user_id', 'book_id', 'is_read', 'rating', 'is_reviewed')
    real_val.write.parquet('hdfs:/user/smt570/' + str(int(percent*100)) + '_val.parquet')
    
    real_test = real_test.select('user_id', 'book_id', 'is_read', 'rating', 'is_reviewed')
    real_test.write.parquet('hdfs:/user/smt570/' + str(int(percent*100)) + '_test.parquet')

    # Write trainvalCOMBO to parquet
    trainvalCOMBO = train_data.union(val_data)
    trainvalCOMBO.write.parquet('hdfs:/user/smt570/' + str(int(percent*100)) + '_trainvalCOMBO.parquet')

    print('Done!')
    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('supervised_train').getOrCreate()
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('percent')
    
    args = parser.parse_args()
    percent = float(args.percent)
    print('Splitting on ' + str(int(percent*100)) + '% of users.')

    # Call our main routine
    main(spark, percent)
