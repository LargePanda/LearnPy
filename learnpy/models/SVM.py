__author__ = 'Jiarui Xu'

from learnpy.models.Model import Model
import pandas as pd

from learnpy.models.ModelUtil import *
import operator
import time
import matplotlib.pyplot as plt

# Here we use SGD for homework purpose
# Ref: https://courses.engr.illinois.edu/cs446/sp2015/Slides/Lecture04.pdf


class SVM(Model):
    def __init__(self, data, class_column, the_num_iter, the_alpha):
        """
        Constructor for SVM model
        :param data:
        :param class_column:
        :param the_num_iter:
        :param the_alpha:
        :return:
        """
        print("Support Vector Machine (SGD version)\n")
        print("Number of Iteration is", the_num_iter, "\n")
        print("Learning Rate is", the_alpha, "\n")

        # set number of iterations (for now)
        self.num_iter = the_num_iter
        self.alpha = the_alpha
        self.weights = None

        # create report and summary
        self.predict_summary = {}
        self.fit_report = {}

        # get the data
        self.data = data
        self.class_column = class_column

        # get the class column and get classes
        col_data = self.data[class_column]
        self.class_list = unique_list(col_data)

        # map class to -1 or 1
        self.class_func = {self.class_list[0]: -1, self.class_list[1]: 1}

        # get the data input {numeric columns, categorical columns}
        self.num_cols, self.cat_cols = get_both_columns(self.data, class_column)

        # only get numeric data
        self.data = numericalize_data(self.data, self.num_cols, self.class_column, self.class_func)
        self.stats_info = stats_info(self.data, self.num_cols, self.class_column, self.class_func)

        # only get numeric data
        self.data = normalize_data(self.data)

        # map 0 to -1, 1 to 1
        self.np_func = {0.0: -1, 1.0: 1}
        self.data[self.class_column] = self.data[self.class_column].map(lambda x: self.np_func[x])

        print(self.stats_info)

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
        feature_set = self.data[self.num_cols].as_matrix()
        label_array = self.data[[self.class_column]].as_matrix()

        # make my uin as seed
        # implement user's ability next week
        np.random.seed(651862182)

        # initialize weights randomly with mean 0
        # feature_set.shape[1]+1 is the number of inputs, plus 1 for the bias term
        syn0 = 2 * np.random.random(feature_set.shape[1]+1) - 1

        for iter in range(0, self.num_iter):
            for i in range(0, feature_set.shape[0]):
                feature = np.append(feature_set[i], 1.0)
                f = np.dot(feature, syn0)
                delta = self.alpha * (label_array[i] - f) * feature
                syn0 += delta

        # save the weight in model weights
        self.weights = syn0
        print(self.weights)

        output = []
        # predict the classes
        for i in range(0, feature_set.shape[0]):
            feature = np.append(feature_set[i], 1.0)
            f = np.dot(feature, syn0)
            output.append(get_class_np(f))

        # output = list(pd.DataFrame(l1)[0].map(lambda x: get_class_np(x)))
        target = list(pd.DataFrame(label_array)[0])

        self.fit_report['output'] = output
        self.fit_report['target'] = target

        print("Fit done.")
        self.report()

    def predict(self, testing_data):
        """
        Predict on testing data
        :param testing_data: testing data
        :return:
        """
        testing_data = transform_data(testing_data, self.num_cols, self.class_column, self.stats_info, self.class_func)
        # print(testing_data)

        feature_set = testing_data[self.num_cols].as_matrix()
        label_array = testing_data[[self.class_column]].as_matrix()


        output = []
        # for each row, calculate dot product and get class
        for i in range(0, feature_set.shape[0]):
            feature = np.append(feature_set[i], 1.0)
            f = np.dot(feature, self.weights)
            output.append(get_class_np(f))

        target = list(pd.DataFrame(label_array)[0])

        self.predict_summary['output'] = output
        self.predict_summary['target'] = target

        print("Fit done.")

        # report the testing summary
        self.summary()

    def summary(self):
        print("The testing accuracy is: ", calculate_acc(self.predict_summary['output'], self.predict_summary['target']))

    def report(self):
        print("The training accuracy is: ", calculate_acc(self.fit_report['output'], self.fit_report['target']))

    def plot_demo(self):
        """
        plot the 2d exmaple for demo purpose
        :return:
        """
        colors = np.where(self.data['Name'] > 0, 'r', 'k')

        # create the scatter plot
        ax = self.data.plot(kind='scatter', x='X', y='Y', c=colors)

        # get the perceptron
        x = np.linspace(-0.2, 1.2, 100) # 100 linearly spaced numbers
        y = (-self.weights[2] - self.weights[0]*x)/self.weights[1] # computing the values of sin(x)/x
        ax.plot(y, x)

        # save figure
        fig = ax.get_figure()
        fig.savefig('svm_figure.png')