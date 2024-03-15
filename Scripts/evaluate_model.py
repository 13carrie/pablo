import os
import sys
# import shap
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession, DataFrame

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
    predictions_df = saved_model.transform(test_df)

    predictions_df.select("Label", "GT", "prediction").show(5)

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
    # print("Acc: {:3f}")
    # print("F1-score: {:3f}")
    # print("Precision: {:3f}")
    # print("Recall: {:3f}")
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
