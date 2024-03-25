import os
import sys
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
    test_df = spark.read.csv(abs_test_dir_path, header=True, inferSchema=True) # recreating test dataframe from test csv
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

    # # get performance metrics of saved model when tested on test_df, using predictions_df
    # metric_start = time.time()
    # get_prediction_metrics(predictions_df)
    # metric_end = time.time() - metric_start
    # print("Time to get performance metrics: ", metric_end)

    # get shap values
    test_df.drop("features")
    print("Beginning SHAP value computation...")
    shap_values = get_shap_values(saved_model, test_df.toPandas())

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
    eval_accuracy = MulticlassClassificationEvaluator(labelCol="GT", predictionCol="prediction",
                                                      metricName="accuracy")
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


# a multi-node implementation for getting shap values
# def multi_get_shap_values(model: PipelineModel, df: DataFrame):
#     return

# a single-node implementation for getting shap values from a pandas df
def get_shap_values(model, test_df):
    # create shap explainer with values
    test_df = test_df.drop(columns=['features']) # features' 'sparsevector' type is not compatible with shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_df, check_additivity=False)
    return shap_values


evaluate_model()



# - **Use grid-search for automatic parameter tuning, or cross-validation (or repeated random samplings) to increase
# the confidence of the results.** The notebook only trains (and tests) an ML saved_model once. The resulting
# performance can be biased (e.g., it can be due to a lucky sampling for train or test). To derive more statistically
# significant results, more trials should be done.
