__author__ = 'Jiarui Xu'

from learnpy.models.Model import Model
import pandas as pd
from learnpy.models.ModelUtil import *
import operator
import time


class NeuralNetwork(Model):
    def __init__(self, data, class_column, the_num_iter):
        """
        Constructor for model Neural Network Model
        :param data: default data
        :param class_column: the label column
        :return:
        """
        print("Neural Network Model created!")
        print("Number of Iteration is", the_num_iter, "\n")

        # set number of iterations (for now)
        self.num_iter = the_num_iter

        # create report and summary
        self.predict_summary = {}
        self.fit_report = {}

        # get the data
        self.data = data
        self.class_column = class_column

        # get the class column and get classes
        col_data = self.data[class_column]
        self.class_list = unique_list(col_data)

        # map class to 0 or 1
        self.class_func = {self.class_list[0]: 0, self.class_list[1]: 1}

        # get the data input {numeric columns, categorical columns}
        self.num_cols, self.cat_cols = get_both_columns(self.data, class_column)

        # only get numeric data
        self.data = numericalize_data(self.data, self.num_cols, self.class_column, self.class_func)
        self.stats_info = stats_info(self.data, self.num_cols, self.class_column, self.class_func)


        # set weights
        self.syn0 = None

        # normalize the data set
        self.data = normalize_data(self.data)
        info_print("Data is normalized. Use fit() to train on the data set.")

    def fit(self, data):
        """
        Fit the model onto a given data set
        :param data:
        :return:
        """
        if data is None:
            self.train_self()
        else:
            # not needed this week
            pass

    def train_self(self):
        """
        Train the model on the data we have in the model
        :return:
        """
        # for this week, we implement 1 layer neural network

        feature_set = self.data[self.num_cols].as_matrix()
        label_array = self.data[[self.class_column]].as_matrix()

        # make my uin as seed
        # implement user's ability next week
        np.random.seed(651862182)

        # initialize weights randomly with mean 0
        # feature_set.shape[1] is the number of inputs
        syn0 = 2 * np.random.random((feature_set.shape[1], 1)) - 1

        for i in range(0, self.num_iter):

            # forward propagation
            l0 = feature_set
            l1 = nonlin(np.dot(l0, syn0))

            # how much did we miss?
            l1_error = label_array - l1

            # multiply how error by the slope of the sigmoid at the values in l1
            l1_delta = l1_error * nonlin(l1, True)

            # update weights
            syn0 += np.dot(l0.T, l1_delta)

        self.syn0 = syn0

        # Output After Training:
        # l1
        output = list(pd.DataFrame(l1)[0].map(lambda x: get_class(x)))
        target = list(pd.DataFrame(label_array)[0])

        self.fit_report['output'] = output
        self.fit_report['target'] = target

        print("Fit done.")
        self.report()

    def predict(self, testing_data):
        """
        Apply the model on the given data
        :param data:
        :return:
        """
        testing_data = transform_data(testing_data, self.num_cols, self.class_column, self.stats_info, self.class_func)

        feature_set = testing_data[self.num_cols].as_matrix()
        label_array = testing_data[[self.class_column]].as_matrix()

        l1 = nonlin(np.dot(feature_set, self.syn0))
        output = list(pd.DataFrame(l1)[0].map(lambda x: get_class(x)))
        target = list(pd.DataFrame(label_array)[0])

        self.predict_summary['output'] = output
        self.predict_summary['target'] = target

        print("Testing done.")
        self.summary()

    def summary(self):
        """
        Summarize the prediction diagnostics
        :return:
        """
        print("The testing accuracy is: ", calculate_acc(self.predict_summary['output'], self.predict_summary['target']))

    def report(self):
        """
        Report the training accuracies
        :return:
        """
        print("The training accuracy is: ", calculate_acc(self.fit_report['output'], self.fit_report['target']))
