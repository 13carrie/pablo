import csv
import os

def convert_dir_of_csvs_to_list(dir_name) -> list:
    csvs_to_convert = []
    abs_path = os.getcwd() + dir_name  # fix this later, creates ~/.../MachineLearningCVE/

    for f in os.listdir(abs_path):
        if f.endswith(".csv"):
            csvs_to_convert.append(f)

    csv_list = []
    for f in csvs_to_convert:
        csv_list.append(convert_csv_to_dict_list((abs_path + f)))
    print("Obtained all CSVs from " + dir_name)
    return csv_list


def convert_csv_to_dict_list(filename) -> list:
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",")
        return list(reader)

