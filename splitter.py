from sklearn.model_selection import train_test_split
import pandas as pd


# splits dataframe into stratified train and test dataframes, protecting the underlying class distribution
# random_state = seed for reproducibility, currently set to 1
def stratified_split(dataframe, test_size):
    return train_test_split(dataframe, dataframe[' Label'], test_size=test_size, random_state=1,
                            stratify=dataframe[' Label'])


# isolates port scan and patator attacks into one dataframe
def port_patator_isolation_split(dataframe, test_size):
    df = pd.DataFrame()
    df.append(isolate_attacks(dataframe, ' Label', ['PortScan', 'FTP-Patator', 'SSH-Patator']))
    return df


# isolates particular attacks from the dataframe
def isolate_attacks(dataframe, column_name, *search_strings):
    matching_rows = dataframe[dataframe[column_name].str.contains('|'.join(search_strings), case=False, na=False)]
    return matching_rows
