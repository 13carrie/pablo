#!/usr/bin/env python
# pablo -- Portscan And Bruteforce Learner Of anomalies, experiment 1
# pablo exp 1 is the first pablo prototype; it trains and tests on CICIDS2017, and utilises SHAP for explainability
# coding: utf-8
# initial loader example by Giovanni Apruzzese from the University of Liechtenstein, 2022
# portscan/bruteforce/patator dataset modification and SHAP explainability by Caroline Smith
# from King's College London, 2024

import time
import numpy as np
import pandas as pd
import os
import sklearn
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, \
    PrecisionRecallDisplay, RocCurveDisplay
from splitter import random_split
from pyspark.mllib.tree import RandomForest
from pyspark.sql import SparkSession
from pyspark.pandas import read_csv
from pyspark import SparkContext, SparkConf
import pyspark.sql.functions as F
from pyspark.sql.types import *

print("scikit-learn version: {}".format(sklearn.__version__))
print("Pandas version: {}".format(pd.__version__))
print("NumPy version: {}".format(np.__version__))

# The code below assumes that you have already downloaded the CICIDS17 dataset in the "machine learning" format,
# and extracted the corresponding archive. The unzipped archive should contain 8 files, each placed into a folder,
# which we will be referred to as "root folder" in this program


# Reading CSV files, and merging all of them into a single DataFrame
spark = SparkSession.builder.getOrCreate()
original_cve = "/MachineLearningCVE/"
root_folder = os.getcwd() + original_cve
df = read_csv(os.path.join(root_folder + "*.csv"))

# PREPROCESSING.
# Deleting rows with NaN/infinite values for a feature
df.replace([np.inf, -np.inf], np.nan, inplace=True)
print("Columns with problematic values: ", list(df.columns[df.isna().any()]))
df.dropna()

# Deleting rows with duplicate values for all features
df.drop_duplicates()

# Deleting duplicate fwd header length column in df
df.drop([' Fwd Header Length.1'])

# Create a new column that unifies all malicious classes into a single class for binary classification
# set benign samples = 1, malicious samples = 0 (modifying categorical 'attack class' feature)
df['GT'] = np.where(df[' Label'] == 'BENIGN', 1, 0)

df.show(5)

# This is the ground truth column. Let's show which classes it contains
print("Unique labels: ", df[' Label'].unique())

# Simple split
train, test = random_split(df, 0.5)

# Define the features used by the classifier, i.e. X
features = pd.Index([' Destination Port', ' Flow Duration', ' Total Fwd Packets',
                     ' Total Backward Packets', 'Total Length of Fwd Packets',
                     ' Total Length of Bwd Packets', ' Fwd Packet Length Max',
                     ' Fwd Packet Length Min', ' Fwd Packet Length Mean',
                     ' Fwd Packet Length Std', 'Bwd Packet Length Max',
                     ' Bwd Packet Length Min', ' Bwd Packet Length Mean',
                     ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s',
                     ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
                     'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max',
                     ' Fwd IAT Min', 'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std',
                     ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', ' Bwd PSH Flags',
                     ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length',
                     ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s',
                     ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean',
                     ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count',
                     ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count',
                     ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count',
                     ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size',
                     ' Avg Fwd Segment Size', ' Avg Bwd Segment Size',
                     'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk',
                     ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk',
                     'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes',
                     ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 'Init_Win_bytes_forward',
                     ' Init_Win_bytes_backward', ' act_data_pkt_fwd',
                     ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max',
                     ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min'])

print("Beginning training...")
start = time.time()
# replicating gints and engelen hyperparams

rfClf_bin = RandomForestClassifier(max_depth=30, n_estimators=100, verbose=True, n_jobs=-1)
rfClf_bin.fit(train[features], train['GT'])
end = time.time() - start
print("Training time: ", end)
predictions_bin = rfClf_bin.predict(test[features])

print("Acc: {:3f}".format(accuracy_score(test['GT'], predictions_bin)))
print("F1-score: {:3f}".format(f1_score(test['GT'], predictions_bin, pos_label='Malicious')))
print("Precision: {:3f}".format(precision_score(test['GT'], predictions_bin, pos_label='Malicious')))
print("Recall: {:3f}".format(recall_score(test['GT'], predictions_bin, pos_label='Malicious')))
print("Matthews Correlation Coefficient (MCC): {:3f}".format(matthews_corrcoef(test['GT'], predictions_bin)))

# rfClf_pr_display = PrecisionRecallDisplay.from_predictions(test['GT'], predictions_bin, pos_label='Malicious')
# rfClf_roc_display = RocCurveDisplay.from_predictions(test['GT'], predictions_bin, pos_label='Malicious')
pd.crosstab(test['GT'], predictions_bin, rownames=['True'], colnames=['Pred'])

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
