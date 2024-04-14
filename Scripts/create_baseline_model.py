#!/usr/bin/env python
# pablo -- Portscan And Bruteforce Learner Of attacks, this creates prototype trained_model for pablo
# coding: utf-8
# loader format by Giovanni Apruzzese from the University of Liechtenstein, 2022
# spark implementation by Caroline Smith
# King's College London, 2024

import os
import time
import sys
import pyspark
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from datetime import datetime
from preprocess_cic import load_data
from split_data import split_dataset

print("Pyspark version: {}".format(pyspark.__version__))

# Creating new Spark Session configured with larger memory space
spark = SparkSession.builder.appName("Pablo Experiment 1")\
    .config("spark.driver.memory", "15g")\
    .getOrCreate()

print("Created new Spark Session")

# Loading preprocessed CSV files, and merging all of them into a single DataFrame
cwd = os.getcwd()
csv_dir = sys.argv[1]
rel_csv_dir_path = "Data/Processed/" + csv_dir
df = load_data(rel_csv_dir_path)



# create train-test split:
training_data, test_data = split_dataset(df, sys.argv)
print("Train/Test data split successful.")

# Define the features used by the classifier for training
training_data = training_data.drop("Label") # drop label column since not numeric
test_data = test_data.drop("Label")  # drop label column since not numeric

col_list = df.columns
features_to_remove = ["Label", "GT"]  # features to not be included in training for model
features = list(set(col_list) - set(features_to_remove))

# use VectorAssembler to add 'features' column to df, containing values of all features used for training
vector_assembler = VectorAssembler(inputCols=features, outputCol="features")
training_data = vector_assembler.transform(training_data)
test_data = vector_assembler.transform(test_data)


# create random forest classifier with hyperparams from engelen et al.
rf = RandomForestClassifier(maxDepth=30, numTrees=100, labelCol="GT", featuresCol="features")

# fit random forest model to training data
print("Beginning training pipeline...")
start = time.time()
rf_model = rf.fit(training_data)
end = time.time() - start
print("Training pipeline time: ", end)

# create unique model id for easy identification
now = datetime.now()
unique_identifier = now.strftime("-%d-%m-%Y-%H%M")

# save model in trained_models dir
model_path = cwd + "/Scripts/Trained_Models/" + sys.argv[1] + unique_identifier
rf_model.save(model_path)
print("Saved ", csv_dir, " model")


test_data = test_data.drop("features") # deleting features col for CSV compatibility

# save test data for later use
test_csv_path = cwd + "/Data/Processed-Test/test_data" + unique_identifier
test_data.write.option("header", True).mode("overwrite").csv(test_csv_path)
print("Saved test data in Data/Processed-Test directory")
