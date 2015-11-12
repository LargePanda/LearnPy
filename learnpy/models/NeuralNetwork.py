__author__ = 'Jiarui Xu'

from learnpy.models.Model import Model
import pandas as pd
from learnpy.models.ModelUtil import *
import operator
import time


class NeuralNetwork(Model):
    def __init__(self, data, class_column):

        # set number of iterations (for now)
        self.num_iter = 300

        # create report
        self.predict_summary = {}
        self.fit_report = {}

        # self.data=data
        self.data = data
        self.class_column = class_column

        # get the class column and get classes
        col_data = self.data[class_column]
        self.class_list = unique_list(col_data)

        # map class to 0 or 1
        class_func = {self.class_list[0]: 0, self.class_list[1]: 1}

        # get the data input
        self.num_cols, self.cat_cols = get_both_columns(self.data, class_column)

        # change data to be used for ANN
        self.data = numericalize_data(self.data, self.num_cols, self.class_column, class_func)

        # normalize the data set
        self.data = normalize_data(self.data)

        # print the data for debugging
        # print(self.data)

        self.fit()
        print("Neural Network Model")

    def fit(self):
        # for this week, we implement 1 layer neural network

        feature_set = self.data[self.num_cols].as_matrix()
        label_array = self.data[[self.class_column]].as_matrix()

        # make my uin as seed
        # implement user's ability next week
        np.random.seed(651862182)

        # initialize weights randomly with mean 0
        syn0 = 2 * np.random.random((feature_set.shape[1], 1)) - 1

        for i in range(0, self.num_iter):

            # forward propagation
            l0 = feature_set
            l1 = nonlin(np.dot(l0, syn0))

            # how much did we miss?
            l1_error = label_array - l1

            # multiply how much we missed by the
            # slope of the sigmoid at the values in l1
            l1_delta = l1_error * nonlin(l1, True)

            # update weights
            syn0 += np.dot(l0.T, l1_delta)

        # Output After Training:
        # l1
        output = list(pd.DataFrame(l1)[0].map(lambda x: get_class(x)))
        target = list(pd.DataFrame(label_array)[0])

        self.fit_report['output'] = output
        self.fit_report['target'] = target

        print("fit done")
        self.report()

    def predict(self, data):
        # task 4.3.3, for next week
        pass

    def summary(self):
        pass

    def report(self):
        print("The accuracy is: ", calculate_acc(self.fit_report['output'], self.fit_report['target']))
