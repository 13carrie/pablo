#!/usr/bin/env python
# coding: utf-8
# initial loader example by Giovanni Apruzzese from the University of Liechtenstein, 2022

import pandas as pd
import numpy as np
import os, time
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

print("scikit-learn version: {}".format(sklearn.__version__))
print("Pandas version: {}".format(pd.__version__))
print("NumPy version: {}".format(np.__version__))

# The code below assumes that you have already downloaded the CICIDS17 dataset in the "machine learning" format,
# and extracted the corresponding archive. The unzipped archive should contain 8 files, each placed into a folder,
# which we will be referred to as "root folder" in this notebook

# TODO: all columns in the CSV should align, data should be complete and valid
# Reading CSV files, and merging all of them into a single DataFrame
root_folder = os.getcwd() + "/MachineLearningCVE/"
df = pd.DataFrame()
for f in os.listdir(root_folder):
    print("Reading: ", f)
    df = df._append(pd.read_csv(os.path.join(root_folder + f)))

# QUICK PREPROCESSING.
# Some classifiers do not like "infinite" (inf) or "null" (NaN) values.
df.replace([np.inf, -np.inf], np.nan, inplace=True)
print("Columns with problematic values: ", list(df.columns[df.isna().any()]))
df.dropna(inplace=True)

# Show all columns (we need to see which column is the 'Ground Truth' of each sample, and which will be the features
# used to describe each sample)
df.columns

# This is the ground truth column. Let's show which classes contains
df[' Label'].unique()

# Create a new column that unifies all malicious classes into a single class for binary classification
df['GT'] = np.where(df[' Label'] == 'BENIGN', 'Benign', 'Malicious')

# Simple split
train, test = train_test_split(df, test_size=0.5)

# Define the features used by the classifier
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
                     ' Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk',
                     ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk',
                     'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes',
                     ' Subflow Bwd Packets', ' Subflow Bwd Bytes', 'Init_Win_bytes_forward',
                     ' Init_Win_bytes_backward', ' act_data_pkt_fwd',
                     ' min_seg_size_forward', 'Active Mean', ' Active Std', ' Active Max',
                     ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min'])

# Train and test a (binary) RandomForestClassifier, printing some basic performance scores, training time,
# and confusion matrix
start = time.time()
rfClf_bin = RandomForestClassifier(n_jobs=-2)
rfClf_bin.fit(train[features], train['GT'])
end = time.time() - start
print("Training time: ", end)
predictions_bin = rfClf_bin.predict(test[features])
print("Acc: {:3f}".format(accuracy_score(test['GT'], predictions_bin)))
print("F1-score: {:3f}".format(f1_score(test['GT'], predictions_bin, pos_label='Malicious')))
pd.crosstab(test['GT'], predictions_bin, rownames=['True'], colnames=['Pred'])

# Train and test a (multiclass) RandomForestClassifier, printing some basic performance scores, training time,
# and confusion matrix
start = time.time()
rfClf_multi = RandomForestClassifier(n_jobs=-2)
rfClf_multi.fit(train[features], train[' Label'])
end = time.time() - start
print("Training time: ", end)
predictions_multi = rfClf_multi.predict(test[features])
print("Acc: {:3f}".format(accuracy_score(test[' Label'], predictions_multi)))
print("F1-score: {:3f}".format(f1_score(test[' Label'], predictions_multi, average='macro')))
pd.crosstab(test[' Label'], predictions_multi, rownames=['True'], colnames=['Pred'])

# **Where to go from here?**
# Here are some ways that can be used to kickstart some research on ML-NIDS by using the code above.
# 
# - **Deal with __inf__ or __NaN__ values.** In the notebook, I removed all of these samples. You may want to keep
# them by, e.g., assigning them a fixed value
# - **Tinker with the features.** In the notebook, I used all features available. Some features may be excessively
# correlated to a given class, which may not be realistic (perhaps a
# rule-based NIDS, instead of an ML one, can be applied to detect that specific attack.) Some may be useless,
# and can be removed. In some cases, some features will be 'categorical', and you must choose how to deal with them (
# e.g., factorize, or onehotencoding).
# - **Change the train:test split.** In the notebook, I simply randomly split
# the initial dataset. You may want to do this on a "class" basis (e.g., take 80% of benign samples and 20% of
# malicious samples for train, and put the rest in test). You may even want to see what happens as fewer data is
# provided in the training set.
# - **Use Validation partition for parameter optimization.** In the notebook,
# I simply split data into train and test, and fed such data to a RandomForestClassifier using default parameters.
# You may want to optimize the performance of such classifier, but to do it fairly you must **not** use the test set.
# Doing this requires to split the train set into two distinct partitions: a "sub_train" and a "validation"
# partition.
# - **Use grid-search for automatic parameter tuning, or cross-validation (or repeated random samplings)
# to increase the confidence of the results.** The notebook only trains (and tests) an ML model once. The resulting
# performance can be biased (e.g., it can be due to a lucky sampling for train or test). To derive more statistically
# significant results, more trials should be done.
# - **Explore different Classifiers and Architectures.** The
# notebook only uses a classifier based on the Random Forest algorithm. There are many more classifiers available on
# scikit-learn. You can even, e.g., devise ensembles of classifiers (consider looking into the [mlxtend](
# http://rasbt.github.io/mlxtend/) library), each focused on a single attack.
# - **Consider deep learning.** The code
# above uses scikit-learn. You can move everything to TensorFlow and use Deep Neural Networks (warning: do this only
# if you have a GPU!) - **Choose a different dataset**. The experiments on this notebook only apply to the CICIDS17
# dataset. Given that network environments are very diverse, I strongly suggest repeating other experiments on a
# different dataset and see if the resulting performance is comparable. Alternatively, you can consider subsets of
# CICIDS17 (e.g., only one day)
# - **Visualizations!** The code above only prints the results and corresponding
# confusion matrix. You may want to visualize the results with proper graphs (via e.g., matplotlib, or seaborn).
# 
# 
# **Tip**: to avoid wasting time, always save your results and also consider saving your ML models (or datasets) as
# pickle files! Nothing is more painful than doing a bunch of experiments and then losing everything!
#
