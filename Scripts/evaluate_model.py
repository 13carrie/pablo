import os
import sys
from collections.abc import Iterator

import numpy as np
import pandas as pd
import shap
import time
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import StructType, StructField, FloatType

spark = SparkSession.builder.appName("Pablo Evaluation") \
    .config("spark.driver.memory", "15g") \
    .getOrCreate()

cwd = os.getcwd()


def evaluate_model():
    # get model information based on provided command line argument
    model_name, abs_model_path, model_id = get_unique_model_info()
    print("Loading saved model...")
    saved_model = RandomForestClassificationModel.load(abs_model_path)

    abs_test_dir_path = get_test_data_directory(model_id)  # try to load test data with filename as [test_data-model_id]
    test_df = spark.read.csv(abs_test_dir_path, header=True,
                             inferSchema=True)  # recreating test dataframe from test csv
    features_to_remove = ["GT"]  # features to not be included in training for model
    col_list = test_df.columns
    features = list(set(col_list) - set(features_to_remove))

    # use VectorAssembler to re-add 'features' column to test_df, containing values of all features used for training
    vector_assembler = VectorAssembler(inputCols=features, outputCol="features")
    test_df = vector_assembler.transform(test_df)
    print("Rows in test dataset: ", test_df.count())

    # get dataframe with additional 'features', 'rawPrediction', 'probability', and 'prediction' columns
    pred_start = time.time()
    predictions_df = saved_model.transform(test_df)
    print()
    pred_end = time.time() - pred_start
    print("Time to get prediction outputs: ", pred_end)
    # print(predictions_df.head())

    # get performance metrics of saved model when tested on test_df, using predictions_df
    # metric_start = time.time()
    # get_prediction_metrics(predictions_df)
    # metric_end = time.time() - metric_start
    # print("Time to get performance metrics: ", metric_end)

    # get shap values
    test_df.drop("features")
    print("Beginning SHAP value computation...")
    test_df = test_df.toPandas()
    shap_values = get_shap_values_multicore(saved_model, test_df)


# returns 'unique' model information
def get_unique_model_info():
    model_name = sys.argv[1]  # retrieve model name stored under trained_models
    model_path = cwd + "/Scripts/Trained_Models/" + model_name  # get absolute path of model
    model_id = model_name[-15:]  # model_id = isolate unique model id from trained_model file name
    return model_name, model_path, model_id


# returns absolute path to test data directory
def get_test_data_directory(model_id):
    test_dir = "/Data/Processed-Test/" + "test_data-" + model_id  # directory of test data file that matches with model
    abs_test_dir_path = cwd + test_dir  # absolute path of test data directory
    return abs_test_dir_path


# prints accuracy, precision, recall, f1, auc scores for model predictions on test data
def get_prediction_metrics(predictions_df: DataFrame):
    # assumption that predictions dataframe contains 'features', 'rawPrediction', 'probability', 'prediction' columns
    eval_accuracy = MulticlassClassificationEvaluator(labelCol="GT", predictionCol="prediction", metricName="accuracy")
    eval_precision = MulticlassClassificationEvaluator(labelCol="GT", predictionCol="prediction",
                                                       metricName="precisionByLabel")
    eval_recall = MulticlassClassificationEvaluator(labelCol="GT", predictionCol="prediction",
                                                    metricName="recallByLabel")
    eval_f1 = MulticlassClassificationEvaluator(labelCol="GT", predictionCol="prediction", metricName="f1")
    eval_auc = BinaryClassificationEvaluator(labelCol="GT", rawPredictionCol="prediction")

    accuracy = eval_accuracy.evaluate(predictions_df)
    f1_score = eval_f1.evaluate(predictions_df)
    precision = eval_precision.evaluate(predictions_df)
    recall = eval_recall.evaluate(predictions_df)
    auc = eval_auc.evaluate(predictions_df)

    print(f"Accuracy: {accuracy}")
    print(f"F1-score: {f1_score}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"AUC (Area Under ROC Curve: {auc}")
    # print("Matthews Correlation Coefficient (MCC): {:3f}")


# the original single-node implementation for getting shap values from a pandas df
def get_shap_values(model, test_df):
    # create shap explainer with values
    test_df = test_df.drop(columns=['features'])  # features' 'sparsevector' type is not compatible with shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_df, check_additivity=False)
    return shap_values


# the multi-node implementation for getting shap values from a pandas df
# calculate_shap udf from https://www.databricks.com/blog/2022/02/02/scaling-shap-calculations-with-pyspark-and-pandas-udf.html
def get_shap_values_multicore(model, test_df):
    start = time.time()
    test_df = test_df.drop(columns=['features'])  # features column 'sparsevector' type is not compatible with shap
    cols = test_df.columns
    print("Creating SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    print("Successfully created SHAP explainer")

    def calculate_shap(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        for X in iterator:
            yield pd.DataFrame(
                explainer.shap_values(np.array(X), check_additivity=False)[0],
                columns=cols,
            )

    return_schema = StructType()
    for feature in cols:
        return_schema = return_schema.add(StructField(feature, FloatType()))
    print("Built return schema")

    spark_test_df = spark.createDataFrame(test_df)
    print("converted df to spark")
    shap_values = spark_test_df.mapInPandas(calculate_shap, schema=return_schema)
    print("Got SHAP values woohoo")
    end = time.time() - start
    print("SHAP computation time: ", end)
    return shap_values


evaluate_model()

