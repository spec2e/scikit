import pandas as pd


def read_wines():
    data = read_and_sample()

    wine_data = get_data(data)
    labels = get_labels(data)
    print(labels)
    return [wine_data, labels]


def read_and_sample():
    # data = pd.read_csv('data/wine_normalized.csv', sep=',', header=None)
    data = pd.read_csv('data/wine.csv', sep=',', header=None)
    sampled = data.sample(frac=1.0)
    sampled.to_csv('data/sampled.csv', sep=',', header=None, index=False)
    data = pd.read_csv('data/sampled.csv', sep=',', header=None)
    return data


def get_labels(wine_data):
    ret_labels = list()
    for index in range(0, len(wine_data)):
        # ret_labels.append(wine_data.ix[index][13])
        ret_labels.append(wine_data.ix[index][0])

    return ret_labels


def get_data(wine_data):
    ret_data = list()
    for x in range(0, len(wine_data)):
        # ret_data.append(wine_data.ix[x][0:-1])
        ret_data.append(wine_data.ix[x][1:])

    return ret_data
