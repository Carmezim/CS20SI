"""
Stanford CS20SI - Assignment #1
Problem 2, Task 3: Logistic regression model to predict whether someone has coronary heart disease
Author: Adriano Carmezim

"""

import tensorflow as tf
import pandas as pd
import numpy as np


DATA_PATH = "../data/"

COLUMNS = ["sbp", "tobacco", "ldl", "adiposity", "famhist", "typea", "obesity",
           "alcohol", "age", "chd"]

FEATURES = ["sbp", "tobacco", "ldl", "adiposity", "famhist", "typea", "obesity",
           "alcohol", "age"]

LABEL = "chd"

training_set = pd.read_csv(DATA_PATH + "heart_train.csv", skipinitialspace=True, skiprows=1, name=COLUMNS)
test_set = pd.read_csv(DATA_PATH + "heart_test.csv", skipinitialspace=True, skiprows=1, name=COLUMNS)
prediction_set = pd.read_csv(DATA_PATH + "heart_pred.csv", skipinitialspace=True, skiprows=1, name=COLUMNS)

feature_cols = [tf.layers.sparse_column_with_keys(column_name=n, keys=[0, 1]) for n in FEATURES]

regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                      hidden_units=[10, 10],
                                      model_dir="/model")

def get_input_fn(data_set, num_epochs=None, shuffle=True):
    return tf.estimator.inputs.pandas_input_fn(
        x = pd.DataFrame({k: data_set[k].values for k in FEATURES}),
        y = pd.Series(data_set[LABEL].value)
    )


X = tf.placeholder([batch_size, 9])
Y = tf.placeholder([batch_size, 1])
