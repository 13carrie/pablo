from sklearn.model_selection import train_test_split

def even_split(dataframe):
    return train_test_split(dataframe, test_size=0.5)
