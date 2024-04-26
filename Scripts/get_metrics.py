import os
from collections.abc import Iterator
import numpy as np
import pandas as pd
import shap
import time
from pyspark.sql.functions import when
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import StructType, StructField, FloatType


spark = SparkSession.builder.appName("Pablo Experiment") \
    .config("spark.driver.memory", "7g") \
    .config("spark.executor.memory", "10g") \
    .config("spark.dynamicAllocation.enabled", "true") \
    .config("spark.executor.cores", 7) \
    .config("spark.default.parallelism", 7) \
    .config("spark.dynamicAllocation.minExecutors", "1") \
    .config("spark.dynamicAllocation.maxExecutors", "7") \
    .getOrCreate()

cwd = os.getcwd()


# returns 'unique' model information, i.e. the identifiers used to reload data
def get_unique_model_info(model_name):
    model_path = cwd + "/Scripts/Trained_Models/" + model_name  # get absolute path of model
    model_id = model_name[-15:]  # model_id = isolate unique model id from trained_model file name
    return model_name, model_path, model_id


# returns absolute path to test data directory
def get_test_data_directory(model_id):
    test_dir = "/Data/Processed-Test/" + "test_data-" + model_id  # directory of test data file that matches with model
    abs_test_dir_path = cwd + test_dir  # absolute path of test data directory
    return abs_test_dir_path


# prints accuracy, precision, recall, f1, auc scores for model predictions on test data
def get_prediction_metrics(predictions_df: DataFrame, label_col: str, prediction_col: str, split: list):
    if len(split) > 2 and split == 'pbp':
        test_labels = [7.0, 10.0, 11.0]
        predictions_df = predictions_df.filter(predictions_df[label_col].isin(test_labels))

    # assumption that predictions dataframe contains 'features', 'rawPrediction', 'probability', 'prediction' columns
    eval_accuracy = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="accuracy")
    accuracy = eval_accuracy.evaluate(predictions_df)

    try:
        eval_precision = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col,
                                                           metricName="precisionByLabel")
        precision = eval_precision.evaluate(predictions_df)
    except Exception as e:
        print("Warning: Precision not able to be evaluated.")
        precision = None
    try:
        eval_recall = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col,
                                                        metricName="recallByLabel")
        recall = eval_recall.evaluate(predictions_df)
    except Exception as e:
        print("Warning: Recall not able to be evaluated.")
        recall = None

    try:
        eval_f1 = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="f1")
        f1_score = eval_f1.evaluate(predictions_df)
    except Exception as e:
        print("Warning: F1 not able to be evaluated.")
        f1_score = None

    try:
        eval_auc_pr = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol=prediction_col,
                                                    metricName="areaUnderPR")
        auc_pr = eval_auc_pr.evaluate(predictions_df)
    except Exception as e:
        print("AUC Pr could not be evaluated")
        auc_pr = None

    print(f"Metrics for comparison of {label_col} to {prediction_col}")
    print(f"Accuracy: {accuracy}")
    print(f"F1-score: {f1_score}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Area Under PR Curve: {auc_pr}")



# the original single-node implementation for getting shap values from a pandas df
def get_shap_values(model, test_df):
    # create shap explainer with values
    # test_df = test_df.drop(columns=['features'])  # features' 'sparsevector' type is not compatible with shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_df, check_additivity=False)  # check_additivity false for pyspark compatibility
    return shap_values, explainer


# the multi-node implementation for getting shap values from a pandas df, not used due to lack of computational power
# calculate_shap udf and four lines below are from https://www.databricks.com/blog/2022/02/02/scaling-shap-calculations-with-pyspark-and-pandas-udf.html
def get_shap_values_multicore(model, test_df):
    start = time.time()
    cols = test_df.columns
    print("Creating SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    print("Successfully created SHAP explainer")

    # udf which calculates shap values in parallel
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

    # must be spark df so that data is compatible with calculate_shap
    spark_test_df = spark.createDataFrame(test_df)
    shap_values = spark_test_df.mapInPandas(calculate_shap, schema=return_schema)
    print("made spark shap_values df")

    # Process SHAP values directly within Spark DataFrame API
    shap_values = shap_values.rdd.map(lambda row: np.array(row[0])).collect()

    # Convert SHAP values to the correct shape
    shap_values = np.array(shap_values)
    shap_values = shap_values.reshape((shap_values.shape[0], len(test_df.columns), -1))

    print("Got SHAP values")
    end = time.time() - start
    print("SHAP computation time: ", end)
    return shap_values, explainer
