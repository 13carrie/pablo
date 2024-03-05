import os
import pandas as pd


def dataframe_converter(csv_name):
    root_folder = os.getcwd() + csv_name
    df = pd.DataFrame()
    for f in os.listdir(root_folder):
        print("Reading: ", f)
        df = df._append(pd.read_csv(os.path.join(root_folder + f)))
    return df
