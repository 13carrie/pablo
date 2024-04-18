import os
import pyspark
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import sys
from pyspark.sql.types import IntegerType

# do all data preprocessing for a csv file and store it in the data/processed directory for easy use by models

print("Pyspark version: {}".format(pyspark.__version__))

spark = SparkSession.builder.appName("Preprocess CICIDS2017 and Improved CICIDS2017 Data") \
    .config("spark.driver.memory", "15g") \
    .config("spark.executor.memory", "5g") \
    .config("spark.dynamicAllocation.enabled", "true") \
    .config("spark.executor.cores", 7) \
    .config("spark.dynamicAllocation.minExecutors", "1") \
    .config("spark.dynamicAllocation.maxExecutors", "7") \
    .config("spark.master", "local") \
    .getOrCreate()


# preprocess a directory of csv files, assuming that said directory is stored in code/data/raw directory
# note that writing your dataframe to disk may automatically partition the dataframe into multiple partial files within
# the new csv_dir directory
def preprocess_data(csv_dir: str):
    rel_csv_path = "/Data/Raw/" + csv_dir
    df = load_data(rel_csv_path)
    df = clean_dataframe(df)
    df = create_ground_truth(df)
    df.printSchema()
    new_csv_path = os.getcwd() + "/Data/Processed/" + csv_dir
    df.write.option("header", True).mode("overwrite").csv(new_csv_path)


# Takes a str directory as a param and returns a Spark DataFrame with supplied headers
def load_data(csv_path: str) -> DataFrame:
    # Loading CSV files, and merging all of them into a single DataFrame
    cwd = os.getcwd()
    abs_path = cwd + "/" + csv_path
    try:
        df = spark.read.csv(abs_path, header=True, inferSchema=True)
    except ValueError:
        print("Failed to read csv")
        raise ValueError

    print("New dataframe created")
    return df


# gets rid of 'bad data' in the dataframe -- cols causing spurious correlations or with bad data/formatting
def clean_dataframe(df: DataFrame) -> DataFrame:
    # Fixes any leading/trailing whitespace in column names
    df = df.select([col(c).alias(c.strip()) for c in df.columns])
    # Deletes rows with NaN/infinite values for a feature
    df = df.drop('Timestamp', 'Dst IP', 'Src IP', 'Flow ID')  # dropping unnecessary columns
    df = df.replace(to_replace=[np.inf, -np.inf], value=None)
    df = df.dropna(how="any")  # Deleting any rows with null/None values
    df = df.filter(col('Label') != 'ATTEMPTED')  # Deleting any rows with 'attempted' attacks, otherwise attempted
    # portscan/patator attacks would be included in training for novel attacks (snooping)
    df = df.drop_duplicates()  # Deleting rows with duplicate values for all features
    return df


# Uses encoding to convert string Label values to integers for compatibility with SHAP and Pyspark
def create_ground_truth(df: DataFrame) -> DataFrame:
    string_indexer = StringIndexer(inputCol="Label", outputCol="GT", stringOrderType="alphabetAsc")
    df = string_indexer.fit(df).transform(df)
    schema_displayer = df.dropDuplicates(['Label'])
    print(schema_displayer.select('Label', 'GT').collect())
    df = df.withColumn("GT", col("GT").cast(IntegerType()))
    return df


preprocess_data(sys.argv[1])
