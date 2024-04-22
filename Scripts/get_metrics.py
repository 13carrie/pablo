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


spark = SparkSession.builder.appName("Pablo Evaluation") \
    .config("spark.driver.memory", "15g") \
    .getOrCreate()

cwd = os.getcwd()


def evaluate_model():
    # get model information based on provided command line argument
    model_arg = sys.argv[1]  # retrieve model name stored under trained_models
    model_name, abs_model_path, model_id = get_unique_model_info(model_arg)
    print("Loading saved model...")
    saved_model = RandomForestClassificationModel.load(abs_model_path)

    abs_test_dir_path = get_test_data_directory(model_id)  # try to load test data with filename as [test_data-model_id]
    test_df = spark.read.csv(abs_test_dir_path, header=True, inferSchema=True)  # recreating test dataframe from test csv
    cols_to_remove = ["GT"]  # features to not be included in testing for model
    col_list = test_df.columns
    features = list(set(col_list) - set(cols_to_remove))

    # use VectorAssembler to re-add 'features' column to test_df, containing values of all features used for training
    vector_assembler = VectorAssembler(inputCols=features, outputCol="features")
    test_df = vector_assembler.transform(test_df)
    print("Rows in test dataset: ", test_df.count())
    test_df.printSchema()

    # get dataframe with additional 'features', 'rawPrediction', 'probability', and 'prediction' columns
    pred_start = time.time()
    predictions_df = saved_model.transform(test_df)
    pred_end = time.time() - pred_start
    print("Time to get prediction outputs: ", pred_end)

    random_df = predictions_df.sample(0.05)
    print(random_df.select("GT", "prediction").show(50))

    # get performance metrics of saved model when tested on test_df, using predictions_dataframe
    # also get binary performance metrics (i.e. was it still able to identify generally if data was benign or malicious?)
    metric_start = time.time()
    get_holistic_prediction_metrics(predictions_df, label_col="GT", prediction_col="prediction")
    metric_end = time.time() - metric_start
    print("Time to get all performance metrics for classification: ", metric_end)

    predictions_csv_path = cwd + "/Data/Results/predictions_data" + model_id
    predictions_df.write.option("header", True).mode("overwrite").csv(predictions_csv_path)


# returns 'unique' model information
def get_unique_model_info(model_name):
    model_path = cwd + "/Scripts/Trained_Models/" + model_name  # get absolute path of model
    model_id = model_name[-15:]  # model_id = isolate unique model id from trained_model file name
    return model_name, model_path, model_id


# returns absolute path to test data directory
def get_test_data_directory(model_id):
    test_dir = "/Data/Processed-Test/" + "test_data-" + model_id  # directory of test data file that matches with model
    abs_test_dir_path = cwd + test_dir  # absolute path of test data directory
    return abs_test_dir_path

# returns both binary and multiclass evaluation of model performance
# i.e. ability of model to correctly classify individual classes
# as well as ability of model to correctly classify traffic generally, either as malicious or benign
def get_holistic_prediction_metrics(predictions_df: DataFrame, label_col: str, prediction_col: str):
    get_prediction_metrics(predictions_df, label_col=label_col, prediction_col=prediction_col)

    binary_predictions_df = convert_to_binary_classification(predictions_df)  # if benign, 0; else 1
    get_prediction_metrics(binary_predictions_df, label_col="GT", prediction_col="prediction")


# prints accuracy, precision, recall, f1, auc scores for model predictions on test data
def get_prediction_metrics(predictions_df: DataFrame, label_col: str, prediction_col: str):
    # assumption that predictions dataframe contains 'features', 'rawPrediction', 'probability', 'prediction' columns
    eval_accuracy = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="accuracy")
    eval_precision = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="precisionByLabel")
    eval_recall = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="recallByLabel")
    eval_f1 = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="f1")
    eval_auc_pr = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol=prediction_col, metricName="areaUnderPR")
    accuracy = eval_accuracy.evaluate(predictions_df)
    f1_score = eval_f1.evaluate(predictions_df)
    precision = eval_precision.evaluate(predictions_df)
    recall = eval_recall.evaluate(predictions_df)
    auc_pr = eval_auc_pr.evaluate(predictions_df)

    print(f"Metrics for comparison of {label_col} to {prediction_col}")
    print(f"Accuracy: {accuracy}")
    print(f"F1-score: {f1_score}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Area Under PR Curve: {auc_pr}")


def convert_to_binary_classification(predictions_df: DataFrame):
    predictions_df = predictions_df.withColumn("GT", when(predictions_df["GT"] == 0.0, 0.0).otherwise(1.0))
    predictions_df = predictions_df.withColumn("prediction", when(predictions_df["prediction"] == 0.0, 0.0).otherwise(1.0))
    return predictions_df


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
    shap_values = np.array(shap_values.collect())  # Extract shap values as array from the dataframe
    shap_values = shap_values.reshape(
        (shap_values.shape[0], len(test_df.columns), -1))  # convert SHAP values to the correct shape

    print("Got SHAP values")
    end = time.time() - start
    print("SHAP computation time: ", end)
    return shap_values, explainer


def plot_shap_force(shap_values, explainer, df, indexes, class_index):
    for index in indexes:
        shap.force_plot(
            explainer.expected_value[class_index], # use expected value for the specified class
            shap_values[index][:, :, class_index],  # extract SHAP values for the specified class
            df.iloc[index],  # original input data for the observation
            show=True
        )
