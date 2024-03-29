from pyspark.sql import SparkSession
from pyspark.sql import DataFrame

spark = SparkSession.builder.appName("Pablo split test 1")\
    .config("spark.driver.memory", "15g")\
    .getOrCreate()

# splits a given dataframe based on an optional keyword
def split_dataset(df: DataFrame, args: list[str]):
    # if no specified split or an invalid specification, do default random split
    # if pbp specified, do pbp isolation split
    if len(args) <= 2:
        print("No split specified, commencing random split 70:30")
        training_data, test_data = df.randomSplit([0.7, 0.3])
    else:
        specified_split = args[2]
        training_data, test_data = match_keyword(df, specified_split)
    return training_data, test_data

# do train/test split based on dataframe and given keyword
def match_keyword(df: DataFrame, keyword: str):
    match keyword:
        case "pbp":
            print("Commencing PBP isolation split as specified")
            training_data, test_data = isolate_attacks(df, ["PortScan", "FTP-Patator", "SSH-Patator"])
        case _:
            print("Commencing random split 70:30")
            training_data, test_data = df.randomSplit([0.7, 0.3])
    return training_data, test_data



# isolates particular attacks from the dataframe
def isolate_attacks(df: DataFrame, search_strings: list):
    # create empty dataframe with same column and column types as test_df argument
    isolated_df = spark.createDataFrame([], schema=df.schema)

    # for each attack you want in isolated_df, append all rows of that attack to isolated_df
    for search_string in search_strings:
        attack_rows = df.filter(df["Label"].contains(search_string)).collect()  # get list of rows containing attack
        attack_df = spark.createDataFrame(data=attack_rows, schema=isolated_df.schema)  # create test df from rows
        isolated_df = isolated_df.union(attack_df)  # add attack_df's contents to isolated_df

    if isolated_df.count() == 0:
        print("No rows containing requested attacks could be found")
        return isolated_df, df
    else:
        # remaining_df (to be used as training df) contains all rows in test_df that are not in the isolated test_df
        remaining_df = df.exceptAll(isolated_df)
        return remaining_df, isolated_df
