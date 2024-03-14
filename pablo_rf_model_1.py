#!/usr/bin/env python
# pablo -- Portscan And Bruteforce Learner Of anomalies, prototype model 1 out of 3
# pablo exp 1 is the first pablo prototype; it trains and tests on the original CICIDS2017 dataset
# coding: utf-8
# initial loader example by Giovanni Apruzzese from the University of Liechtenstein, 2022
# spark implementation, portscan/bruteforce/patator dataset modification, and SHAP explainability by Caroline Smith
# King's College London, 2024
import os
import time
import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import *

print("Pyspark version: {}".format(pyspark.__version__))

# Creating new Spark Session configured with larger memory space
spark = SparkSession.builder.appName("Pablo Experiment 1")\
    .config("spark.driver.memory", "15g")\
    .getOrCreate()

print("Created new Spark Session")

# Reading CSV files, and merging all of them into a single DataFrame
original_csvs = "/MachineLearningCVE/"
cwd = os.getcwd()
abs_path = cwd + original_csvs
df = spark.read.csv(abs_path, header=True, inferSchema=True)

print("New dataframe created")

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
print("Train/Test data split successful.")

print("Beginning training pipeline...")
start = time.time()
rf_model = pipeline.fit(training_data)

# replicating gints and engelen hyperparams
end = time.time() - start
print("Training pipeline time: ", end)

print("Saving model...")
model_path = cwd + "/pablo_model_1"
rf_model.save(model_path)