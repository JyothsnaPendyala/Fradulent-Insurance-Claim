import pandas as pd

def extract_data():
    data = pd.read_csv('train.csv')
    print(data.head())
    return data

extract_data()