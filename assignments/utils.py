import pandas as pd
import numpy as np
import os

class data_preprocess:

    def read_data(file_path, delimiter=';'):
        # try:
        #     os.path.exists(file_path)
        # except OSError:
        #     pass
        print(file_path)
        data = pd.read_csv(str(file_path), delimiter=delimiter, header=0)
        return data

    def encode(data=None, column_name=None):
        column_name = str(column_name)

        # generate indicators from categorical data
        indicators = pd.get_dummies(data[column_name], columns=column_name, prefix=column_name, drop_first=False)

        # replace original categorical values for numerical representation
        data[column_name] = indicators[column_name + "_Present"]
        return data

    def scale_data(data=None, columns=None):
        for column in columns:
            data[column] = (data[column] - data[column].min()) / data[column].max()

        return data

    def split_data(data=None, split_fraction=0.7):
        labels = data["chd"]
        features = data.drop(["chd"], axis=1)
        features, labels = np.array(features), np.array(labels)

        n_samples = len(features)
        split_i = int(split_fraction*n_samples)
        train_X, train_Y = features[:split_i], labels[:split_i]
        test_X, test_Y = features[split_i:], labels[split_i:]

        return train_X, train_Y, test_X, test_Y, features, labels



    def check_data_sample(data):
        print(data.head())

