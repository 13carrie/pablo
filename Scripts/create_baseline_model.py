#!/usr/bin/env python
# pablo -- Portscan And Bruteforce Learner Of attacks, this creates prototype trained_model for pablo
# coding: utf-8
# loader format by Giovanni Apruzzese from the University of Liechtenstein, 2022
# spark implementation by Caroline Smith
# King's College London, 2024

import os
import time
import sys
import numpy as np
import pyspark
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import VectorAssembler
from datetime import datetime
from preprocess_cic import load_data
from split_data import split_dataset, isolate_gt_label, get_row_with_matching_cols_index
from get_metrics import get_shap_values_multicore, plot_shap_force, get_prediction_metrics

print("Pyspark version: {}".format(pyspark.__version__))

# Creating new Spark Session configured with larger memory space
spark = SparkSession.builder.appName("Pablo Experiment 1") \
    .config("spark.driver.memory", "15g") \
    .config("spark.executor.memory", "5g") \
    .config("spark.dynamicAllocation.enabled", "true") \
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
training_data = training_data.drop("Label")  # drop label column since not numeric
test_data = test_data.drop("Label")  # drop label column since not numeric

col_list = df.columns
columns_to_exclude = ["Label", "GT"]  # features to not be included in training for model
features = list(set(col_list) - set(columns_to_exclude))

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

# get dataframe with additional 'features', 'rawPrediction', 'probability', and 'prediction' columns
pred_start = time.time()
predictions_df = rf_model.transform(test_data)
pred_end = time.time() - pred_start
print("Time to get prediction outputs: ", pred_end)


# get performance metrics of saved model when tested on test_data, using predictions_dataframe
metric_start = time.time()
get_prediction_metrics(predictions_df, label_col="GT", prediction_col="prediction")
metric_end = time.time() - metric_start
print("Time to get performance metrics for total classification: ", metric_end)

# get SHAP visualisations for portscan and patator local explanations:

# step 1: prepare X (predictions_df) and y (gt_array)
gt_array = np.array(test_data.select("GT").collect())
pd_test_df = test_data.toPandas().drop(columns=['features', 'GT'])  # features column 'sparsevector' type is not compatible with shap

# step 2: get shap values
shap_values, explainer = get_shap_values_multicore(rf_model, pd_test_df)
print(type(shap_values))
shap_values.info()
shap_values.head()
# step 3: get row indexes for observations to plot
# portscan_index = get_row_with_matching_cols_index(test_data, 10.0, "GT", "prediction")
# ssh_index = get_row_with_matching_cols_index(test_data, 11.0, "GT", "prediction")
# ftp_index = get_row_with_matching_cols_index(test_data, 7.0, "GT", "prediction")
#
# # step 4: make plots
# indexes = [portscan_index, ssh_index, ftp_index]
# plot_shap_force(shap_values, explainer, pd_test_df, indexes)
#
# # save model in trained_models dir
# now = datetime.now()
# unique_identifier = now.strftime("-%d-%m-%Y-%H%M")  # create unique model id for easy identification
# model_path = cwd + "/Scripts/Trained_Models/" + sys.argv[1] + unique_identifier
# rf_model.save(model_path)
# print("Saved ", csv_dir, " model")
#
# # save test data in test_data dir for later use
# test_data = test_data.drop("features")  # deleting features col for CSV compatibility
# test_csv_path = cwd + "/Data/Processed-Test/test_data" + unique_identifier
# test_data.write.option("header", True).mode("overwrite").csv(test_csv_path)
# print("Saved test data in Data/Processed-Test directory")
#
# # save predictions in results dir for later shap use
# predictions_df = predictions_df.drop("features", "rawPrediction", "probability") # deleting cols incompatible with CSV format
# preds_path = cwd + "/Data/Results/pred_" + unique_identifier
# predictions_df.write.option("header", True).mode("overwrite").json(preds_path)
# print("saved prediction data in Data/Results")
