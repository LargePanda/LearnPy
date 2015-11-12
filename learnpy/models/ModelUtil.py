__author__ = 'Jiarui Xu'
import pandas as pd
import math
import numpy as np


def unique_list(coldata):
    """
    Get a list of unique elements
    :param coldata: a column of data
    :return: a list of unique elements
    """
    return list(set(coldata))


def get_both_columns(data, class_column):
    print("Response Variable is ", "[", class_column, "]")
    cols = data.columns
    num_cols = list(set(data._get_numeric_data().columns) - set(class_column))
    cat_cols = list(set(cols) - set(num_cols))
    cat_cols.remove(class_column)
    return num_cols, cat_cols


def get_mean_std(coldata):
    return coldata.mean(), coldata.std()


def calculate_prob(x, mean, std):
    exponent = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(std, 2))))
    return (1 / (math.sqrt(2*math.pi) * std)) * exponent


# sigmoid function
def nonlin(x, deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))


def numericalize_data(data, num_cols, class_column, class_func):
    num_data = data[num_cols]
    num_data.is_copy = False
    num_data[class_column] = data[class_column].map(lambda x: class_func[x])
    return num_data


def normalize_data(df):
    data = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    return data


def get_class(x):
    if x<0.5:
        return 0
    else:
        return 1


def calculate_acc(y, y0):
    m = 0
    for i in range(0, len(y0)):
        if y[i] == y0[i]:
            m += 1
    return float(m)/len(y0)


