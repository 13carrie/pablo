from pyspark.sql import SparkSession
from pyspark.sql import DataFrame

spark = SparkSession.builder.appName("Pablo Experiment") \
    .config("spark.driver.memory", "7g") \
    .config("spark.executor.memory", "10g") \
    .config("spark.dynamicAllocation.enabled", "true") \
    .config("spark.executor.cores", 7) \
    .config("spark.default.parallelism", 7) \
    .config("spark.dynamicAllocation.minExecutors", "1") \
    .config("spark.dynamicAllocation.maxExecutors", "7") \
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
        print(specified_split)
        training_data, test_data = match_keyword(df, specified_split)
    return training_data, test_data


# do train/test split based on dataframe and given keyword
def match_keyword(df: DataFrame, keyword: str):
    match keyword:
        case "pbp":
            print("Commencing PBP isolation split as specified")
            training_data, test_data = isolate_attacks(df, ["10.0", "11.0", "7.0"])
        case '' | _:
            print("Commencing random split 75:25")
            training_data, test_data = df.randomSplit([0.75, 0.25])
    return training_data, test_data


# isolates particular attacks from the dataframe
def isolate_attacks(df: DataFrame, search_values: list):
    # create empty dataframe with same column and column types as test_df argument
    isolated_df = spark.createDataFrame([], schema=df.schema)

    # for each attack you want in isolated_df, append all rows of that attack to isolated_df
    for search_value in search_values:
        print(search_value)
        attack_rows = df.filter(df["GT"] == search_value).collect()  # get list of rows containing attack
        attack_df = spark.createDataFrame(data=attack_rows, schema=isolated_df.schema)  # create test df from rows
        print("attack_df length for ", search_value, ": ", attack_df.count())
        isolated_df = isolated_df.union(attack_df)  # add attack_df's contents to isolated_df
        print("current isolated df length: ", isolated_df.count())

    if isolated_df.count() == 0:
        print("No rows containing requested attacks could be found")
        return df, isolated_df
    else:
        # remaining_df (to be used as training df) contains all rows in test_df that are not in the isolated test_df
        remaining_df = df.exceptAll(isolated_df)
        print("assert isolated_df contains GT 10: ", isolated_df.filter(isolated_df['GT'] == 10.0).count())
        print("assert isolated_df contains GT 11: ", isolated_df.filter(isolated_df['GT'] == 11.0).count())
        print("assert isolated_df contains GT 7: ", isolated_df.filter(isolated_df['GT'] == 7.0).count())
        print(isolated_df.show(10))
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

# similar to above function, but returns the row rather than the index
def get_row_with_matching_cols(df: DataFrame, val, col_1: str, col_2: str):
    return_row = None
    df_collect = df.collect()
    for row in df_collect:
        if (row.__getitem__(col_1) == val) and (row.__getitem__(col_2) == val):
            return_row = row
            break
    if return_row is None:
        print("row not found for these values and columns")
    return return_row

# similar to above function, but only returns row based on value of one column
def get_gt_row(df: DataFrame, val: float, col: str):
    return_row = None
    df_collect = df.collect()
    for row in df_collect:
        if row.__getitem__(col) == val:
            return_row = row
            break
    if return_row is None:
        print("row not found for GT of value: ", val)
    return return_row
