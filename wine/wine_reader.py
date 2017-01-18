import pandas as pd


def read_wines():
    data = read()

    wine_data = get_data(data)
    labels = get_labels(data)
    return [wine_data, labels]


def read():
    data = pd.read_csv('data/wine_normalized.csv', sep=',', header=None)
    return data


def get_labels(wine_data):
    ret_labels = list()
    for index in range(0, len(wine_data)):
        ret_labels.append(wine_data.ix[index][13])
        #ret_labels.append(wine_data.ix[index][0])

    return ret_labels


def get_data(wine_data):
    ret_data = list()
    for x in range(0, len(wine_data)):
        ret_data.append(wine_data.ix[x][0:-1])

    return ret_data
