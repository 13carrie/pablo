import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType
import sys

# do all data preprocessing for a csv file and store it in the data/processed directory for easy use by models

print("Pyspark version: {}".format(pyspark.__version__))

spark = SparkSession.builder.appName("Preprocess CICIDS2017 and Improved CICIDS2017 Data") \
    .config("spark.driver.memory", "15g") \
    .config("spark.master", "local") \
    .getOrCreate()


# preprocess a directory of csv files, assuming that said directory is stored in code/data/raw directory
# note that writing your dataframe to disk may automatically partition the dataframe into multiple partial files within
# the new csv_dir directory
def preprocess_data():
    csv_dir = sys.argv[1]
    rel_csv_path = "/Data/Raw/" + csv_dir
    df = load_data(rel_csv_path)
    df = clean_dataframe(df)
    df = create_ground_truth(df)
    df.printSchema()
    new_csv_path = os.getcwd() + "/Data/Processed/" + csv_dir
    df.write.option("header", True).mode("overwrite").csv(new_csv_path)



# Takes a path to a directory as a param and returns a Spark DataFrame with supplied headers
def load_data(csv_path: str) -> DataFrame:
    # Loading CSV files, and merging all of them into a single DataFrame
    cwd = os.getcwd()
    abs_path = cwd + "/" + csv_path
    df = spark.read.csv(abs_path, header=True, inferSchema=True)

    print("New dataframe created")
    return df


# Deletes rows with NaN/infinite values for a feature
def clean_dataframe(df: DataFrame) -> DataFrame:
    df = df.replace(to_replace=[np.inf, -np.inf], value=None)
    df = df.dropna(how="any")  # Deleting any rows with null/None values
    df = df.drop_duplicates()  # Deleting rows with duplicate values for all features
    return df


# Uses binary encoding to convert labels to 1 (malicious) or 0 (benign) values
def create_ground_truth(df: DataFrame) -> DataFrame:
    df = df.withColumn(" GT", when(df[" Label"] == 'BENIGN', '1').otherwise('0'))
    df = df.withColumn(" GT", col(" GT").cast(IntegerType()))
    return df


preprocess_data()
