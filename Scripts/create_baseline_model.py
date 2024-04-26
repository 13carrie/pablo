#!/usr/bin/env python
# pablo -- Portscan And Bruteforce Learner Of attacks
# This file creates experiment models 1.1, 1.2 and 2.1
# command for exp 1.1: python3 Scripts/create_baseline_model.py MachineLearningCVE
# command for exp 1.2: python3 Scripts/create_baseline_model.py ImprovedMachineLearningCVE
# command for exp 2.1: python3 Scripts/create_baseline_model.py ImprovedMachineLearningCVE pbp
# coding: utf-8
# author: Caroline Smith
# King's College London, 2024

import os
import time
import sys
from datetime import datetime
import pyspark
import shap
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from preprocess_cic import load_data
from split_data import split_dataset, get_row_with_matching_cols, get_gt_row
from get_metrics import get_shap_values, get_prediction_metrics

print("Pyspark version: {}".format(pyspark.__version__))

# Creating new Spark Session configured with larger memory space
spark = SparkSession.builder.appName("Pablo Experiment") \
    .config("spark.driver.memory", "7g") \
    .config("spark.executor.memory", "10g") \
    .config("spark.dynamicAllocation.enabled", "true") \
    .config("spark.executor.cores", 7) \
    .config("spark.default.parallelism", 7) \
    .config("spark.dynamicAllocation.minExecutors", "1") \
    .config("spark.dynamicAllocation.maxExecutors", "7") \
    .getOrCreate()

print("Created new Spark Session")

# Loading preprocessed CSV files, and merging all of them into a single DataFrame
cwd = os.getcwd()
args = sys.argv
csv_dir = sys.argv[1]

if len(args) > 2 and args[2] == 'pbp':
    is_pbp = True
else:
    is_pbp = False
rel_csv_dir_path = "Data/Processed/" + csv_dir

df = load_data(rel_csv_dir_path)

# create train-test split:
training_data, test_data = split_dataset(df, args)
print("assert test contains GT 10: ", test_data.filter(test_data['GT'] == 10.0).count())
print("assert test contains GT 11: ", test_data.filter(test_data['GT'] == 11.0).count())
print("assert test contains GT 7: ", test_data.filter(test_data['GT'] == 7.0).count())

print("Train/Test data split successful.")

# Define the features used by the classifier for training
training_data = training_data.drop("Label")  # drop label column since not numeric
test_data = test_data.drop("Label")  # drop label column since not numeric
col_list = df.columns
columns_to_exclude = ["Label", "GT"]  # features to not be included in training for model (avoids spur. correlations)
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

# get dataframe with additional 'rawPrediction', 'probability', and 'prediction' columns
pred_start = time.time()
predictions_df = rf_model.transform(test_data)
print(predictions_df.select("GT", "prediction").show(25))
pred_end = time.time() - pred_start
print("Time to get prediction outputs: ", pred_end)

# get performance metrics of saved model when tested on test_data, using predictions_dataframe
metric_start = time.time()
get_prediction_metrics(predictions_df, label_col="GT", prediction_col="prediction", split=args)
metric_end = time.time() - metric_start
print("Time to get performance metrics for total classification: ", metric_end)

# step 1: create dataframe containing selected observations to plot
if is_pbp:  # likely no correctly predicted rows, instead plot any portscan/ssh/ftp row
    print("pbp selected, doing get_gt_row now")
    portscan_row = get_gt_row(predictions_df, 10.0, "GT")
    ssh_row = get_gt_row(predictions_df, 11.0, "GT")
    ftp_row = get_gt_row(predictions_df, 7.0, "GT")
else:  # likely high-performance, can be trusted to have correctly predicted rows
    print("regular situation")
    portscan_row = get_row_with_matching_cols(predictions_df, 10.0, "GT", "prediction")
    ssh_row = get_row_with_matching_cols(predictions_df, 11.0, "GT", "prediction")
    ftp_row = get_row_with_matching_cols(predictions_df, 7.0, "GT", "prediction")

rows = [portscan_row, ssh_row, ftp_row]
selected_rows_df = spark.createDataFrame(rows)  # make df containing only the observations you want to plot
pd_df = selected_rows_df.drop('features', 'GT', 'rawPrediction',
                              'probability').toPandas()  # make SHAP-compatible pandas version of df

# step 2: get shap values
# shap_values, explainer = get_shap_values_multicore(rf_model, pd_df)
shap_values, explainer = get_shap_values(rf_model, pd_df)

# step 3: make plots according to attack class
class_names = [10, 11, 7]

print("portscan plot: ")
index = 0
shap.force_plot(explainer.expected_value[0],
                shap_values[0][index],
                pd_df.iloc[index, :], matplotlib=True)

print("ssh plot: ")
index = 1
shap.force_plot(explainer.expected_value[1],
                shap_values[1][index],
                pd_df.iloc[index, :], matplotlib=True)

print("ftp plot: ")
index = 2
shap.force_plot(explainer.expected_value[2],
                shap_values[2][index],
                pd_df.iloc[index, :], matplotlib=True)

# save model in trained_models dir
now = datetime.now()
unique_identifier = now.strftime("-%d-%m-%Y-%H%M")  # create unique model id for easy identification
model_path = cwd + "/Scripts/Trained_Models/" + sys.argv[1] + unique_identifier
rf_model.save(model_path)
print("Saved ", csv_dir, " model")

# save test data in test_data dir for later use
test_data = test_data.drop("features")  # deleting features col for CSV compatibility
test_csv_path = cwd + "/Data/Processed-Test/test_data" + unique_identifier
test_data.write.option("header", True).mode("overwrite").csv(test_csv_path)
print("Saved test data in Data/Processed-Test directory")
