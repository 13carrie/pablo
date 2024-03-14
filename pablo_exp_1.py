#!/usr/bin/env python
# pablo -- Portscan And Bruteforce Learner Of anomalies, experiment 1
# pablo exp 1 is the first pablo prototype; it trains and tests on CICIDS2017, and utilises SHAP for explainability
# coding: utf-8
# initial loader example by Giovanni Apruzzese from the University of Liechtenstein, 2022
# spark implementation, portscan/bruteforce/patator dataset modification, and SHAP explainability by Caroline Smith
# King's College London, 2024
import os
import time
import pyspark
import shap
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import *

print("Pyspark version: {}".format(pyspark.__version__))

# The code below assumes that you have already downloaded the CICIDS17 dataset in the "machine learning" format,
# and extracted the corresponding archive. The unzipped archive should contain 8 files, each placed into a folder,
# which we will be referred to as "root folder" in this program



# Creating new Spark Session configured with larger memory space
spark = SparkSession.builder.appName("Pablo Experiment 1")\
    .config("spark.driver.memory", "15g")\
    .getOrCreate()

# Reading CSV files, and merging all of them into a single DataFrame
original_csvs = "/MachineLearningCVE/"
abs_path = os.getcwd() + original_csvs
df = spark.read.csv(abs_path, header=True, inferSchema=True)

# PREPROCESSING:
# Deleting rows with NaN/infinite values for a feature
df = df.replace(to_replace=[np.inf, -np.inf], value=None)
df = df.dropna(how="any")  # Deleting any rows with null/None values
df = df.drop_duplicates()  # Deleting rows with duplicate values for all features

# Create a new column GT that encodes all malicious classes into a single class for binary classification
# set benign samples = 1, malicious samples = 0
df = df.withColumn(" Label", when(df[" Label"] == 'BENIGN', 'Benign').otherwise('Malicious'))
stage_1 = StringIndexer(inputCol=" Label", outputCol=" GT") # create ground truth column

# Define the features used by the classifier
features = [' Destination Port', ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets',
            'Total Length of Fwd Packets', ' Total Length of Bwd Packets', ' Fwd Packet Length Max',
            ' Fwd Packet Length Min', ' Fwd Packet Length Mean', ' Fwd Packet Length Std', 'Bwd Packet Length Max',
            ' Bwd Packet Length Min', ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s',
            ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Total',
            ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean',
            ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags', ' Fwd URG Flags',
            ' Bwd URG Flags', ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s',
            ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std',
            ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count',
            ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio',
            ' Average Packet Size', ' Avg Fwd Segment Size', ' Avg Bwd Segment Size', 'Fwd Avg Bytes/Bulk',
            ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk',
            'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes', ' Subflow Bwd Packets',
            ' Subflow Bwd Bytes', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward', ' act_data_pkt_fwd',
            ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max', ' Active Min', 'Idle Mean',
            ' Idle Std', ' Idle Max', ' Idle Min']
stage_2 = VectorAssembler(inputCols=features, outputCol="features")

# create random forest classifier with hyperparams of gints + engelen's work
stage_3 = RandomForestClassifier(maxDepth=30, numTrees=100, labelCol=" GT", featuresCol="features")

pipeline = Pipeline(stages=[stage_1, stage_2, stage_3])

training_data, test_data = df.randomSplit([0.7, 0.3])

print("Beginning training pipeline...")
start = time.time()
rf_model = pipeline.fit(training_data)

# replicating gints and engelen hyperparams
end = time.time() - start
print("Training pipeline time: ", end)

print("Beginning model testing...")
prediction = rf_model.transform(test_data)

# predictions_bin = rfClf_bin.predict(test[features])

# print("Acc: {:3f}".format(accuracy_score(test['GT'], predictions_bin)))
# print("F1-score: {:3f}".format(f1_score(test['GT'], predictions_bin, pos_label='Malicious')))
# print("Precision: {:3f}".format(precision_score(test['GT'], predictions_bin, pos_label='Malicious')))
# print("Recall: {:3f}".format(recall_score(test['GT'], predictions_bin, pos_label='Malicious')))
# print("Matthews Correlation Coefficient (MCC): {:3f}".format(matthews_corrcoef(test['GT'], predictions_bin)))

# rfClf_pr_display = PrecisionRecallDisplay.from_predictions(test['GT'], predictions_bin, pos_label='Malicious')
# rfClf_roc_display = RocCurveDisplay.from_predictions(test['GT'], predictions_bin, pos_label='Malicious')
# pd.crosstab(test['GT'], predictions_bin, rownames=['True'], colnames=['Pred'])

# compute SHAP values
# print("Beginning SHAP value computation...")
# start = time.time()

# explainer = shap.TreeExplainer(rfClf_bin)
# shap_values = explainer(train[features], check_additivity=False)

# force plot for SHAP values
# shap.plots.force(shap_values[0:100])
# end = time.time() - start
# print("SHAP computation time: ", end)

# **Where to go from here?**
# - **Tinker with the features.** In the notebook, I used all features available. Some features may be excessively
# correlated to a given class, which may not be realistic. Some may be useless,
# and can be removed. In some cases, some features will be 'categorical', and you must choose how to deal with them (
# e.g., factorize, or onehotencoding).
# - **Use grid-search for automatic parameter tuning, or cross-validation (or repeated random samplings)
# to increase the confidence of the results.** The notebook only trains (and tests) an ML model once. The resulting
# performance can be biased (e.g., it can be due to a lucky sampling for train or test). To derive more statistically
# significant results, more trials should be done.
# - **Visualizations!** The code above only prints the results and corresponding
# confusion matrix. You may want to visualize the results with proper graphs (via e.g., matplotlib, or seaborn).
# **Tip**: to avoid wasting time, always save your results and also consider saving your ML models (or datasets) as
# pickle files! Nothing is more painful than doing a bunch of experiments and then losing everything!
