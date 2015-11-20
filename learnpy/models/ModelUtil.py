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
    """
    Get numeric and categorical columns
    :param data: data set
    :param class_column: class column
    :return:
    """
    print("Response Variable is ", "[", class_column, "]")

    # Get the column set
    cols = data.columns

    num_cols = list(set(data._get_numeric_data().columns) - set(class_column))
    cat_cols = list(set(cols) - set(num_cols))

    # excluding class column
    cat_cols.remove(class_column)
    return num_cols, cat_cols


def get_mean_std(coldata):
    """
    Use the pandas builtin function to get mean and std
    :param coldata: column data
    :return:
    """
    return coldata.mean(), coldata.std()


def calculate_prob(x, mean, std):
    """
    calculate Gaussian Probability
    :param x: value
    :param mean: mean
    :param std: standard deviation
    :return: Gaussian Probability
    """
    exponent = math.exp(-(math.pow(x-mean, 2)/(2*math.pow(std, 2))))
    return (1 / (math.sqrt(2*math.pi) * std)) * exponent



def nonlin(x, deriv=False):
    """
    Sigmoid Function
    :param x: value
    :param deriv: whether or not it is derivative
    :return:
    """
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))


def numericalize_data(data, num_cols, class_column, class_func):
    """
    Transform the class data into 0 and 1
    :param data: data set
    :param num_cols: number of columns
    :param class_column: class column
    :param class_func: map from class into 0 or 1
    :return:
    """
    num_data = data[num_cols]
    num_data.is_copy = False
    num_data[class_column] = data[class_column].map(lambda x: class_func[x])
    return num_data


def normalize_data(df):
    """
    Normalize the data set
    :param df: data frame
    :return:
    """
    data = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    return data


def get_class(x):
    """
    Classify based on probability. (0 if p<0.5)
    :param x: probability
    :return:
    """
    if x<0.5:
        return 0
    else:
        return 1


def get_class_np(x):
    """
    Classify based on probability. (-1 if p<0.0)
    :param x: probability
    :return:
    """
    if x<0:
        return -1
    else:
        return 1


def calculate_acc(y, y0):
    """
    Calculate the accuracy based on the actual class list and predicted class list
    :param y:
    :param y0:
    :return:
    """
    m = 0
    for i in range(0, len(y0)):
        if y[i] == y0[i]:
            m += 1
    return float(m)/len(y0)


def info_print(text):
    """
    Helper function to print information
    :param text:
    :return:
    """
    print("[INFO]", " ", text)


def stats_info(data, num_cols):
    """
    Get min max information form the data set
    :param data:
    :param num_cols:
    :return:
    """
    info = {}
    for col in num_cols:
        info[col] = {}
        info[col]["max"] = data[col].max()
        info[col]["min"] = data[col].min()
    return info


def trans_func(x, column, info):
    """
    single function to normalize the columns by min, max
    :param x: value
    :param column: column name
    :param info: min max dict
    :return:
    """
    return (x-info[column]["min"])/(info[column]["max"] - info[column]["min"])


def transform_data(data, num_cols, class_column, info, class_func):
    """
    transform the data into normalized one
    :param data: data set object
    :param num_cols: numerical columns
    :param class_column: class column
    :param info: min max dict
    :param class_func: transform function
    :return:
    """
    for col in num_cols:
        data[col] = data[col].map(lambda x: trans_func(x, col, info))
    data[class_column] = data[class_column].map(lambda x: class_func[x])
    return data
