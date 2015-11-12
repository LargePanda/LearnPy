__author__ = 'Jiarui Xu'

from learnpy.models.Model import Model
import pandas as pd
from learnpy.models.ModelUtil import *
import operator
import time


# to be implemented next week

class NaiveBayes(Model):
    def __init__(self, data, class_column):

        # create report
        self.predict_summary = None
        self.fit_report = None

        # self.data=data
        self.data = data
        self.class_column = class_column

        # get the class column and get classes
        col_data = self.data[class_column]
        self.class_list = unique_list(col_data)

        self.num_cols, self.cat_cols = get_both_columns(self.data, class_column)

        self.prob_hub = {}
        self.fit(self.data)
        res = self.predict(self.data)
        print("Naive Bayes Model")

    def predict(self, data, golden):
        result = []
        for r in range(0, len(data)):
            label = self.predict_single(data.iloc[r])
            result.append(label)

        if golden is not None:
            match = 0
            for i in range(0, len(result)):
                if result[i] == golden[i]:
                    match += 1
            acc = match/len(result)
            print(acc)
        return result

    def predict_single(self, line):
        #print(line)
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
            for claz in self.class_list:
                mean, std = self.prob_hub[col][claz]
                prob_list[claz] *= calculate_prob(val, mean, std)

        return max(prob_list.items(), key=operator.itemgetter(1))[0]



    def fit(self, data):
        # for each numeric column, we need to record mean and std for both classes
        for col in self.num_cols:
            self.prob_hub[col] = {}
            for claz in self.class_list:
                mean, std = get_mean_std(self.data[self.data[self.class_column] == claz][col])
                self.prob_hub[col][claz] = (mean, std)

        # for each categorical columns, we need to record P(X=x|Y=y)
        for col in self.cat_cols:
            print(col)
            ulist = unique_list(self.data[col])
            self.prob_hub[col] = {}
            stat = self.data.groupby(self.class_column)[col].value_counts() / self.data.groupby(self.class_column)[col].count()
            print(stat)
            for claz in self.class_list:
                self.prob_hub[col][claz] = {}
                for uni_element in ulist:
                    self.prob_hub[col][claz][uni_element] = stat[claz][uni_element]

    def summary(self):
        pass

    def report(self):
        pass

    def data_info(self):
        pass


