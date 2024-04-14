#!/usr/bin/env python
# pablo -- Portscan And Bruteforce Learner Of attacks, this creates prototype trained_model for pablo
# coding: utf-8
# original loader format by Giovanni Apruzzese from the University of Liechtenstein, 2022
# spark implementation by Caroline Smith
# King's College London, 2024

import os
import time
import sys
import pyspark
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from datetime import datetime
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from split_data import split_dataset

print("Pyspark version: {}".format(pyspark.__version__))

# Creating new Spark Session configured with larger memory space and dynamic allocation of executors
spark = SparkSession.builder.appName("Pablo Experiment Final") \
    .config("spark.driver.memory", "15g") \
    .config("spark.executor.memory", "5g") \
    .config("spark.dynamicAllocation.enabled", "true") \
    .config("spark.executor.cores", 7) \
    .config("spark.dynamicAllocation.minExecutors", "1") \
    .config("spark.dynamicAllocation.maxExecutors", "7") \
    .getOrCreate()
print("Created new Spark Session")

# Loading preprocessed CSV files, and merging all of them into a single DataFrame
csv_dir = sys.argv[1]
csv_dir_path = "/Data/Processed/" + csv_dir
cwd = os.getcwd()
abs_path = cwd + csv_dir_path

# attempting to get dataframe from specified csv
try:
    df = spark.read.csv(abs_path, header=True, inferSchema=True)
except ValueError:
    print("Failed to read csv, please ensure that data has been processed and stored in Data/Processed directory.")
    raise ValueError
print("New dataframe created")

# create train-test split:
training_data, test_data = split_dataset(df, sys.argv)
print("Train/Test data split successful.")

# Define the features used by the classifier for training
training_data = training_data.drop("Label")  # drop label column since not numeric
test_data = test_data.drop("Label")  # drop label column since not numeric

col_list = df.columns
features_to_remove = ["Label", "GT"]  # features to not be included in training for model
features = list(set(col_list) - set(features_to_remove))

# use VectorAssembler to add 'features' column to df, containing values of all features used for training
vector_assembler = VectorAssembler(inputCols=features, outputCol="features")
training_data = vector_assembler.transform(training_data)
test_data = vector_assembler.transform(test_data)


# create random forest classifier with unspecified hyperparams
rf = RandomForestClassifier(labelCol="GT", featuresCol="features")

# separate train into train and validation set:
# first create param grid for hyperparameter options
param_grid = ParamGridBuilder() \
    .addGrid(rf.maxDepth, [10, 20, 30]) \
    .addGrid(rf.numTrees, [50, 100, 150]) \
    .build()
# then create train validation split from parameter grid and rf classifier
evaluator = BinaryClassificationEvaluator().setLabelCol("GT")  # label column must always be set for evaluator
tvs = TrainValidationSplit(estimator=rf,
                           estimatorParamMaps=param_grid,
                           evaluator=evaluator,
                           trainRatio=0.75,
                           parallelism=5)

# fit train validation model to training data
print("Beginning training pipeline...")
start = time.time()
rf_model = tvs.fit(training_data)
best_model = rf_model.bestModel
end = time.time() - start
print("Training pipeline time: ", end)

# print hyperparams for ideal model
print("Best numTrees value: ", best_model.getNumTrees)
print("Best maxDepth value: ", best_model.getMaxDepth())

# create unique model id for easy identification
now = datetime.now()
unique_identifier = now.strftime("-%d-%m-%Y-%H%M")

# save model in trained_models dir
model_path = cwd + "/Scripts/Trained_Models/" + sys.argv[1] + unique_identifier
best_model.save(model_path)
print("Saved ", csv_dir, " model")

test_data = test_data.drop("features")  # deleting features col for CSV compatibility
# test_data = test_data.withColumn("features", vector_to_array(col("features")))

# save test data for later use
test_csv_path = cwd + "/Data/Processed-Test/test_data" + unique_identifier
test_data.write.option("header", True).mode("overwrite").csv(test_csv_path)
print("Saved test data in Data/Processed-Test directory")
