from sklearn.model_selection import train_test_split

def random_split(dataframe, test_size):
    return train_test_split(dataframe, test_size=test_size)

def stratified_split(dataframe, train_size, test_size):
    return

def isolate_attack(dataframe, attack_name):
    return