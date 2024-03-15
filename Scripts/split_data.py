import pandas as pd

# TODO: adapt these functions for spark


# isolates port scan and patator attacks into one dataframe
def port_patator_isolation_split(dataframe):
    df = pd.DataFrame()
    df.append(isolate_attacks(dataframe, ' Label', ['PortScan', 'FTP-Patator', 'SSH-Patator']))
    return df


# isolates particular attacks from the dataframe
def isolate_attacks(dataframe, column_name, *search_strings):
    matching_rows = dataframe[dataframe[column_name].str.contains('|'.join(search_strings), case=False, na=False)]
    return matching_rows
