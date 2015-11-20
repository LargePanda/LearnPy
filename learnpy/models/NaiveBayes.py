__author__ = 'Jiarui Xu'

from learnpy.models.Model import Model
import pandas as pd
from learnpy.models.ModelUtil import *
import operator
import time


# to be implemented next week

class NaiveBayes(Model):
    def __init__(self, data, class_column):
        """
        Constructor for the Naive Bayes Model
        :param data:
        :param class_column:
        :return:
        """
        print("Naive Bayes Model created!")

        # create report
        self.predict_summary = {}
        self.fit_report = {}

        # self.data=data
        self.data = data
        self.class_column = class_column

        # get the class column and get classes
        col_data = self.data[class_column]
        self.class_list = unique_list(col_data)

        # get numeric columns and categorical columns
        self.num_cols, self.cat_cols = get_both_columns(self.data, class_column)

        # Build the pro
        self.prob_hub = {}

    def predict(self, data, training_report=False):
        """
        Predict the labels for a given data set
        :param data:
        :param golden:
        :return:
        """
        output = []
        for r in range(0, len(data)):
            label = self.predict_single(data.iloc[r])
            output.append(label)

        label_array = data[[self.class_column]].as_matrix()
        target = list(pd.DataFrame(label_array)[0])

        if training_report:
            self.fit_report['output'] = output
            self.fit_report['target'] = target
            print("Training Fit done.")
            self.report()

        else:
            self.predict_summary['output'] = output
            self.predict_summary['target'] = target

            print("Testing done.")
            self.summary()

    def predict_single(self, line):
        """
        Predict the label on a data row
        :param line: a data row
        :return:
        """
        # print(line)
        prob_list = {}
        for claz in self.class_list:
            prob_list[claz] = 1

        # for each cat column
        for col in self.cat_cols:
            val = line[col]
            for claz in self.class_list:
                prob_list[claz] *= self.prob_hub[col][claz][val]

        # for each num column
        for col in self.num_cols:
            val = line[col]
            # for each class
            for claz in self.class_list:
                mean, std = self.prob_hub[col][claz]
                prob_list[claz] *= calculate_prob(val, mean, std)

        return max(prob_list.items(), key=operator.itemgetter(1))[0]

    def fit(self, data):
        """
        Fit the model onto the given data set.
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
        Train the model on the data set in the model
        :return:
        """
        # for each numeric column, we need to record mean and std for both classes
        for col in self.num_cols:
            self.prob_hub[col] = {}
            for claz in self.class_list:
                mean, std = get_mean_std(self.data[self.data[self.class_column] == claz][col])
                self.prob_hub[col][claz] = (mean, std)

        # for each categorical columns, we need to record P(X=x|Y=y)
        for col in self.cat_cols:
            ulist = unique_list(self.data[col])
            self.prob_hub[col] = {}
            stat = self.data.groupby(self.class_column)[col].value_counts() / self.data.groupby(self.class_column)[col].count()
            # for each class
            for claz in self.class_list:
                self.prob_hub[col][claz] = {}
                for uni_element in ulist:
                    self.prob_hub[col][claz][uni_element] = stat[claz][uni_element]

        self.predict(self.data, True)

    def summary(self):
        print("The testing accuracy is: ", calculate_acc(self.predict_summary['output'], self.predict_summary['target']))

    def report(self):
        print("The training accuracy is: ", calculate_acc(self.fit_report['output'], self.fit_report['target']))

    def data_info(self):
        print(self.prob_hub)


