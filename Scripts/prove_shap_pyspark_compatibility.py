import os
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler
import shap

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('test shap').getOrCreate()

# create df and read
cwd = os.getcwd()
abs_path = cwd + "/Data/Processed/ImprovedMachineLearningCVE"
# attempting to get test_df from specified csv
df = spark.read.csv(abs_path, header=True, inferSchema=True)
df = spark.createDataFrame(df.take(100000))

# Define the features used by the classifier for training
df = df.drop('Label')
col_list = df.columns
features_to_remove = ['GT']
features = list(set(col_list) - set(features_to_remove))

# use VectorAssembler to add 'features' column to df, containing values of all features used for training
stage_2 = VectorAssembler(inputCols=features, outputCol="features")
df = stage_2.transform(df)
print("transformed dataframe for features")

# split into train/test dfs
train, test = df.randomSplit([0.7, 0.3], seed=2024)
print("successfully split data")

rf = RandomForestClassifier(featuresCol="features", labelCol="GT")
rfModel = rf.fit(train)
print("trained model")
predictions = rfModel.transform(test)
print("tested model")

# preprocessing for shap
test_pandas = test.toPandas().drop(columns=['features']) # deleted column since type == sparsevector, not compatible

print(test_pandas.head())

# create shap explainer with values
explainer = shap.TreeExplainer(rfModel)
shap_values = explainer.shap_values(test_pandas, check_additivity=False)
