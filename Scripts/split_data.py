from pyspark.sql import SparkSession
from pyspark.sql import DataFrame

spark = SparkSession.builder.appName("Pablo split test 1") \
    .config("spark.driver.memory", "15g") \
    .getOrCreate()


# splits a given dataframe based on an optional keyword
def split_dataset(df: DataFrame, args: list[str]):
    # if no specified split or an invalid specification, do default random split
    # if pbp specified, do pbp isolation split
    if len(args) <= 2:
        print("No split specified, commencing random split 75:25")
        training_data, test_data = df.randomSplit([0.75, 0.25])
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
        case '' | _:
            print("Commencing random split 75:25")
            training_data, test_data = df.randomSplit([0.75, 0.25])
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
        return df, isolated_df
    else:
        # remaining_df (to be used as training df) contains all rows in test_df that are not in the isolated test_df
        remaining_df = df.exceptAll(isolated_df)
        return remaining_df, isolated_df


# returns two dataframes:
# isolated_df contains the data where the double GT label == the specified double attack_label
# remaining_df contains all other data
def isolate_gt_label(df: DataFrame, attack_label):
    isolated_df = df.filter(df["GT"] == attack_label)  # get df containing only specified attack

    if isolated_df.count() == 0:
        print("No rows containing requested attacks could be found")
        return df, isolated_df
    else:
        # remaining_df (to be used as training df) contains all rows in test_df that are not in the isolated test_df
        remaining_df = df.exceptAll(isolated_df)
        return remaining_df, isolated_df


# get index of random row of a pyspark dataframe where two columns are equivalent to a passed value
def get_row_with_matching_cols_index(df: DataFrame, val, col_1: str, col_2: str):
    index = None
    df_collect = df.collect()
    for row in df_collect:
        if (row.__getitem__(col_1) == val) and (row.__getitem__(col_2) == val):
            index = df_collect.index(row)
            break
    if index is None:
        print("index not found for these values and columns")
    return index

def get_row_with_matching_cols(df: DataFrame, val, col_1: str, col_2: str):
    return_row = None
    df_collect = df.collect()
    for row in df_collect:
        if (row.__getitem__(col_1) == val) and (row.__getitem__(col_2) == val):
            return_row = row
            break
    if return_row is None:
        print("index not found for these values and columns")
    return return_row
