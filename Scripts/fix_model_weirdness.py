import os
import time
import sys
import pyspark
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import VectorAssembler
from datetime import datetime
from preprocess_cic import load_data
from split_data import split_dataset

print("Pyspark version: {}".format(pyspark.__version__))

# Creating new Spark Session configured with larger memory space
spark = SparkSession.builder.appName("Pablo Experiment 1")\
    .config("spark.driver.memory", "15g") \
    .config("spark.executor.memory", "5g") \
    .config("spark.dynamicAllocation.enabled", "true") \
    .getOrCreate()

print("Created new Spark Session")

def get_prediction_metrics(predictions_dataframe: DataFrame, label_col: str, prediction_col: str):
    # assumption that predictions dataframe contains 'features', 'rawPrediction', 'probability', 'prediction' columns
    eval_accuracy = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="accuracy")
    eval_precision = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col,
                                                       metricName="precisionByLabel")
    eval_recall = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col,
                                                    metricName="recallByLabel")
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

# Loading preprocessed CSV files, and merging all of them into a single DataFrame
cwd = os.getcwd()
csv_dir = sys.argv[1]
rel_csv_dir_path = "Data/Processed/" + csv_dir
df = load_data(rel_csv_dir_path)

training_data, test_data = split_dataset(df, sys.argv)
print("split train, test dataset")

# Define the features used by the classifier for training
training_data = training_data.drop("Label") # drop label column since not numeric
test_data = test_data.drop("Label")  # drop label column since not numeric

col_list = df.columns
columns_to_exclude = ["Label", "GT"]  # features to not be included in training for model
features = list(set(col_list) - set(columns_to_exclude))

# use VectorAssembler to add 'features' column to df, containing values of all features used for training
vector_assembler = VectorAssembler(inputCols=features, outputCol="features")
training_data = vector_assembler.transform(training_data)
test_data = vector_assembler.transform(test_data)

rf = RandomForestClassifier(maxDepth=30, numTrees=100, labelCol="GT", featuresCol="features")

# fit random forest model to training data
print("Beginning training pipeline...")
start = time.time()
rf_model = rf.fit(training_data)
end = time.time() - start
print("Training pipeline time: ", end)

now = datetime.now()
unique_identifier = now.strftime("-%d-%m-%Y-%H%M")
model_path = cwd + "/Scripts/Trained_Models/" + sys.argv[1] + unique_identifier

rf_model.save(model_path)
rf_model = RandomForestClassificationModel.load(model_path)

# save test data for later use
test_data = test_data.drop("features") # deleting features col for CSV compatibility
test_csv_path = cwd + "/Data/Processed-Test/test_data" + unique_identifier
test_data.write.option("header", True).mode("overwrite").csv(test_csv_path)
print("Saved test data in Data/Processed-Test directory")
test_df = spark.read.csv(test_csv_path, header=True, inferSchema=True)  # recreating test dataframe from test csv

cols_to_remove = ["GT"]  # features to not be included in testing for model
col_list = test_data.columns
features = list(set(col_list) - set(cols_to_remove))

# get dataframe with additional 'features', 'rawPrediction', 'probability', and 'prediction' columns
vector_assembler = VectorAssembler(inputCols=features, outputCol="features")
test_df = vector_assembler.transform(test_df)
pred_start = time.time()
predictions_df = rf_model.transform(test_data)
pred_end = time.time() - pred_start
print("Time to get prediction outputs: ", pred_end)

random_df = predictions_df.sample(0.05)
print(random_df.select("GT", "prediction").show(50))

# get performance metrics of saved model when tested on test_df, using predictions_dataframe
metric_start = time.time()
get_prediction_metrics(predictions_df, label_col="GT", prediction_col="prediction")
metric_end = time.time() - metric_start
print("Time to get performance metrics for total classification: ", metric_end)



