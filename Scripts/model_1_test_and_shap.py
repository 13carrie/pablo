import shap
from pyspark.ml.classification import RandomForestClassificationModel

model = RandomForestClassificationModel.load("pablo_model_1")


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
# **Tip**: to avoid wasting time, always save your results and also consider saving your ML models (or datasets) as
# pickle files! Nothing is more painful than doing a bunch of experiments and then losing everything!
