import os
import sys
from collections.abc import Iterator
import numpy as np
import pandas as pd
import shap
import time
from pyspark.sql.functions import when, col
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import StructType, StructField, FloatType, DoubleType
from split_data import isolate_gt_label
from get_performance_metrics import get_unique_model_info

spark = SparkSession.builder.appName("Pablo SHAP Evaluation") \
    .config("spark.driver.memory", "15g") \
    .getOrCreate()

cwd = os.getcwd()

def get_shap():
    model_arg = sys.argv[1]  # retrieve model name stored under trained_models
    model_name, abs_model_path, model_id = get_unique_model_info(model_arg)
    print("Loading saved model...")
    saved_model = RandomForestClassificationModel.load(abs_model_path)

    abs_preds_dir_path = get_predictions_data_directory(model_id)  # try to load test data with filename as [test_data-model_id]
    pred_df = spark.read.csv(abs_preds_dir_path, header=True, inferSchema=True)  # recreating test dataframe from test csv


    print("Beginning SHAP value computation...")
    pred_df = pred_df.toPandas()
    shap_values = get_shap_values_multicore(saved_model, pred_df)

    visualise_shap(shap_values)

    other_shap_values, portscan_shap_values = isolate_gt_label(shap_values, 10.0)
    other_shap_values, ssh_patator_shap_values = isolate_gt_label(other_shap_values, 11.0)
    other_shap_values, ftp_patator_shap_values = isolate_gt_label(other_shap_values, 7.0)
    patator_shap_values = ssh_patator_shap_values.union(ftp_patator_shap_values)
    portscan_shap_values.write.option("header", True).mode("overwrite").csv((cwd + "/Data/Results/portscan_shap" + model_id))
    patator_shap_values.write.option("header", True).mode("overwrite").csv((cwd + "/Data/Results/patator_shap" + model_id))


# returns absolute path to test data directory
def get_predictions_data_directory(model_id):
    test_dir = "/Data/Results/predictions_data" + model_id  # directory of test data file that matches with model
    abs_preds_dir_path = cwd + test_dir  # absolute path of test data directory
    return abs_preds_dir_path


# the original single-node implementation for getting shap values from a pandas df, not used due to lack of efficiency
def get_shap_values(model, test_df):
    # create shap explainer with values
    test_df = test_df.drop(columns=['features'])  # features' 'sparsevector' type is not compatible with shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_df, check_additivity=False)
    return shap_values


# the multi-node implementation for getting shap values from a pandas df
# calculate_shap udf and four lines below are from https://www.databricks.com/blog/2022/02/02/scaling-shap-calculations-with-pyspark-and-pandas-udf.html
def get_shap_values_multicore(model, test_df):
    start = time.time()
    test_df = test_df.drop(columns=['features'])  # features column 'sparsevector' type is not compatible with shap
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
    print("converted df to spark")
    shap_values = spark_test_df.mapInPandas(calculate_shap, schema=return_schema)
    print("Got SHAP values")
    end = time.time() - start
    print("SHAP computation time: ", end)
    return shap_values

def visualise_shap(shap_values):
    return



