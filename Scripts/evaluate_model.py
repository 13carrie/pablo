import os
import sys
# import shap
import time
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName("Pablo Evaluation 1") \
    .config("spark.driver.memory", "15g") \
    .getOrCreate()

cwd = os.getcwd()


def evaluate_model():
    # get model information based on provided command line argument
    model_name, abs_model_path, model_id = get_unique_model_info()
    saved_model = PipelineModel.load(abs_model_path)

    # try to load test data, where test data in format test_data- + model_id
    abs_test_dir_path = get_test_data_directory(model_id)

    # get test and predicted outcome dataframes
    test_df = spark.read.csv(abs_test_dir_path, header=True, inferSchema=True) # recreating test dataframe from test csv

    # get dataframe with additional 'features', 'rawPrediction', 'probability', and 'prediction' columns
    pred_start = time.time()
    predictions_df = saved_model.transform(test_df)
    pred_end = time.time() - pred_start
    print("Time to get prediction outputs: ", pred_end)

    # get performance metrics of saved model when tested on test_df, using predictions_df
    metric_start = time.time()
    get_prediction_metrics(predictions_df)
    metric_end = time.time() - metric_start
    print("Time to get performance metrics: ", metric_end)
def get_unique_model_info():
    model_name = sys.argv[1]  # retrieve model name stored under trained_models
    model_path = cwd + "/Scripts/Trained_Models/" + model_name  # get absolute path of model
    model_id = model_name.replace('pablo_model_1-', '')  # model_id = remove 'pablo_model_1- from model_name
    return model_name, model_path, model_id

def get_test_data_directory(model_id):
    test_dir = "/Data/Processed-Test/" + "test_data-" + model_id  # directory of test data file that matches with model
    abs_test_dir_path = cwd + test_dir  # absolute path of test data directory
    return abs_test_dir_path


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
    return


def get_shap_values():
    return

evaluate_model()


# compute SHAP values
# print("Beginning SHAP value computation...")
# start = time.time()

# explainer = shap.TreeExplainer(rfClf_bin)
# shap_values = explainer(train[features], check_additivity=False)

# force plot for SHAP values
# shap.plots.force(shap_values[0:100])
# end = time.time() - start
# print("SHAP computation time: ", end)


# - **Use grid-search for automatic parameter tuning, or cross-validation (or repeated random samplings) to increase
# the confidence of the results.** The notebook only trains (and tests) an ML saved_model once. The resulting
# performance can be biased (e.g., it can be due to a lucky sampling for train or test). To derive more statistically
# significant results, more trials should be done.
