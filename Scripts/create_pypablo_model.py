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
import shap
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from datetime import datetime
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from preprocess_cic import load_data
from split_data import split_dataset, get_gt_row
from get_metrics import get_shap_values, get_prediction_metrics

print("Pyspark version: {}".format(pyspark.__version__))

# Creating new Spark Session configured with larger memory space
spark = SparkSession.builder.appName("Pablo Experiment") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "11g") \
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

rel_csv_dir_path = "Data/Processed/" + csv_dir

df = load_data(rel_csv_dir_path)


training_data, test_data = split_dataset(df, args)
print("assert test contains GT 10: ", test_data.filter(test_data['GT'] == 10.0).count())
print("assert test contains GT 11: ", test_data.filter(test_data['GT'] == 11.0).count())
print("assert test contains GT 7: ", test_data.filter(test_data['GT'] == 7.0).count())

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
evaluator = MulticlassClassificationEvaluator().setLabelCol("GT")  # label column must always be set for evaluator
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

# get dataframe with additional 'rawPrediction', 'probability', and 'prediction' columns
pred_start = time.time()
predictions_df = best_model.transform(test_data)
print(predictions_df.select("GT", "prediction").show(25))
pred_end = time.time() - pred_start
print("Time to get prediction outputs: ", pred_end)

# get performance metrics of saved model when tested on test_data, using predictions_dataframe
metric_start = time.time()
get_prediction_metrics(predictions_df, label_col="GT", prediction_col="prediction", split=args)
metric_end = time.time() - metric_start
print("Time to get performance metrics for total classification: ", metric_end)

# step 1: create dataframe containing selected observations to plot
portscan_row = get_gt_row(predictions_df, 10.0, "GT")
ssh_row = get_gt_row(predictions_df, 11.0, "GT")
ftp_row = get_gt_row(predictions_df, 7.0, "GT")

rows = [portscan_row, ssh_row, ftp_row]
selected_rows_df = spark.createDataFrame(rows)  # make df containing only the observations you want to plot
pd_df = selected_rows_df.drop('features', 'GT', 'rawPrediction',
                              'probability').toPandas()  # make SHAP-compatible pandas version of df

# step 2: get shap values
shap_values, explainer = get_shap_values(best_model, pd_df)

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
