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
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import VectorAssembler
from datetime import datetime
from preprocess_cic import load_data
from split_data import split_dataset

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
print("Num 0: ", training_data.filter(df.GT == 0.0).count())
print("Num 1: ", training_data.filter(df.GT == 1.0).count())
print("Num 2: ", training_data.filter(df.GT == 2.0).count())
print("Num 3: ", training_data.filter(df.GT == 3.0).count())
print("Num 4: ", training_data.filter(df.GT == 4.0).count())
print("Num 5: ", training_data.filter(df.GT == 5.0).count())
print("Num 6: ", training_data.filter(df.GT == 6.0).count())
print("Num 7: ", training_data.filter(df.GT == 7.0).count())
print("Num 8: ", training_data.filter(df.GT == 8.0).count())
print("Num 9: ", training_data.filter(df.GT == 9.0).count())
print("Num 10: ", training_data.filter(df.GT == 10.0).count())
print("Num 11: ", training_data.filter(df.GT == 11.0).count())
print("Num 12: ", training_data.filter(df.GT == 12.0).count())
print("Num 13: ", training_data.filter(df.GT == 13.0).count())
print("Num 14: ", training_data.filter(df.GT == 14.0).count())
test_data = vector_assembler.transform(test_data)
print("Num 0: ", test_data.filter(df.GT == 0.0).count())
print("Num 1: ", test_data.filter(df.GT == 1.0).count())
print("Num 2: ", test_data.filter(df.GT == 2.0).count())
print("Num 3: ", test_data.filter(df.GT == 3.0).count())
print("Num 4: ", test_data.filter(df.GT == 4.0).count())
print("Num 5: ", test_data.filter(df.GT == 5.0).count())
print("Num 6: ", test_data.filter(df.GT == 6.0).count())
print("Num 7: ", test_data.filter(df.GT == 7.0).count())
print("Num 8: ", test_data.filter(df.GT == 8.0).count())
print("Num 9: ", test_data.filter(df.GT == 9.0).count())
print("Num 10: ", test_data.filter(df.GT == 10.0).count())
print("Num 11: ", test_data.filter(df.GT == 11.0).count())
print("Num 12: ", test_data.filter(df.GT == 12.0).count())
print("Num 13: ", test_data.filter(df.GT == 13.0).count())
print("Num 14: ", test_data.filter(df.GT == 14.0).count())

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
print("Num 0: ", predictions_df.filter(df.GT == 0.0).count())
print("Num 1: ", predictions_df.filter(df.GT == 1.0).count())
print("Num 2: ", predictions_df.filter(df.GT == 2.0).count())
print("Num 3: ", predictions_df.filter(df.GT == 3.0).count())
print("Num 4: ", predictions_df.filter(df.GT == 4.0).count())
print("Num 5: ", predictions_df.filter(df.GT == 5.0).count())
print("Num 6: ", predictions_df.filter(df.GT == 6.0).count())
print("Num 7: ", predictions_df.filter(df.GT == 7.0).count())
print("Num 8: ", predictions_df.filter(df.GT == 8.0).count())
print("Num 9: ", predictions_df.filter(df.GT == 9.0).count())
print("Num 10: ", predictions_df.filter(df.GT == 10.0).count())
print("Num 11: ", predictions_df.filter(df.GT == 11.0).count())
print("Num 12: ", predictions_df.filter(df.GT == 12.0).count())
print("Num 13: ", predictions_df.filter(df.GT == 13.0).count())
print("Num 14: ", predictions_df.filter(df.GT == 14.0).count())
pred_end = time.time() - pred_start
print("Time to get prediction outputs: ", pred_end)

def get_prediction_metrics(predictions_dataframe: DataFrame, label_col: str, prediction_col: str):
    # assumption that predictions dataframe contains 'features', 'rawPrediction', 'probability', 'prediction' columns
    eval_accuracy = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="accuracy")
    eval_precision = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="precisionByLabel")
    eval_recall = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="recallByLabel")
    eval_f1 = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="f1")
    eval_auc_roc = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol=prediction_col, metricName="areaUnderROC")
    eval_auc_pr = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol=prediction_col, metricName="areaUnderPR")

    accuracy = eval_accuracy.evaluate(predictions_dataframe)
    f1_score = eval_f1.evaluate(predictions_dataframe)
    precision = eval_precision.evaluate(predictions_dataframe)
    recall = eval_recall.evaluate(predictions_dataframe)
    auc_roc = eval_auc_roc.evaluate(predictions_dataframe)
    auc_pr = eval_auc_pr.evaluate(predictions_dataframe)

    print(f"Metrics for comparison of {label_col} to {prediction_col}")
    print(f"Accuracy: {accuracy}")
    print(f"F1-score: {f1_score}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Area Under ROC Curve: {auc_roc}")
    print(f"Area Under PR Curve: {auc_pr}")


# get performance metrics of saved model when tested on test_data, using predictions_dataframe
metric_start = time.time()
get_prediction_metrics(predictions_df, label_col="GT", prediction_col="prediction")
metric_end = time.time() - metric_start
print("Time to get performance metrics for total classification: ", metric_end)

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

# save predictions in results dir for later shap use
predictions_df = predictions_df.drop("features") # deleting features col for CSV compatibility
preds_path = cwd + "/Data/Results/pred_" + unique_identifier
predictions_df.write.option("header", True).mode("overwrite").csv(preds_path)
print("saved prediction data in Data/Results")
