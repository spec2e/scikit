import pandas as pd


def read_wines():
    df = pd.read_csv('wine.data', sep=',', header=None)
    print(df.shape)
    return df
