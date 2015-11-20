__author__ = 'Jiarui Xu'

import pandas as pd
import json as json
from learnpy.models.NeuralNetwork import NeuralNetwork
from learnpy.models.NaiveBayes import NaiveBayes
from learnpy.models.SVM import SVM


class Problem:
    """ A problem is a definition of a machine learning problem.

    """
    def __init__(self, problem_type=None, file_path=None, model_label=None):
        """ Init function for Problem
        :param problem_type: it should be one of the defined types
        :param file_path: a file path to read data from
        :param model: chosen model
        :return:
        """

        # define problem type
        self.problem_type = problem_type
        if problem_type is not None:
            self.check_type()

        # define data set
        self.file_path = file_path
        self.data = None
        if file_path is not None:
            self.read_data()

        # define label
        self.label = None
        if self.label is not None:
            self.check_label()

        # define model
        self.model_label = model_label
        self.model = None
        if model_label is not None:
            self.check_model()
            self.create_model()

        # placeholder for testing data
        self.testing_data = None

    def check_type(self):
        """ Check if problem type is defined
        :param self:
        :return:
        """
        if self.problem_type not in ['BinaryClassification', 'MultiClassification']:
            raise Exception('Problem Type Not Included.')

    def check_label(self):
        """ Check if label is defined
        :param self:
        :return:
        """
        stat = pd.DataFrame.describe(self.data[self.label])
        if(self.problem_type == 'BinaryClassification' and stat['unique'] != 2) \
                or (self.problem_type == 'MultiClassification' and stat['unique'] <= 2):
            raise Exception("Number of unique items is not consistent with problem type")

    def check_model(self):
        """ Check if model is defined
        :param self:
        :return:
        """
        if self.model_label not in ['SVM', 'NaiveBayes', 'NeuralNetwork']:
            raise Exception('Model is not defined')

    def set_model(self, model, num_iter=1000, alpha=0.003):
        """ Set the problem's model
        :param self:
        :param model: model to be set
        :return:
        """
        self.model_label = model
        self.check_model()
        self.create_model(num_iter, alpha)

    def read_data(self):
        """ Read data from the file_path
        :param self:
        :return:
        """
        try:
            self.data = pd.read_csv(self.file_path)
        except IOError:
            raise

    def set_label(self, label):
        """ Set the problem's label
        :param self:
        :param label: label to be set
        :return:
        """
        if label in self.data.columns:
            self.label = label
            self.data[label] = self.data[label].astype(str)
        else:
            raise Exception("Label column does not exist.")

    def save(self, save_path, data_path):
        """ Set the problem's elements
        :param self:
        :param save_path: file path to save info
        :param data_path: file path to save data
        :return:
        """
        problem = dict()
        problem["problem_type"] = self.problem_type
        problem["label"] = self.label
        problem["model_label"] = self.model_label

        self.data.to_json(data_path)
        print("Saved the data to ", data_path)

        with open(save_path, 'w') as fp:
            json.dump(problem, fp, indent=1)
        print("Saved the problem to ", save_path)

    def load(self, load_path, data_path):
        """ Load problem's elements from a file & load problem's data from a file
        :param self:
        :param load_path: file path to load info
        :param data_path: file path to load data
        :return:
        """
        # load path into dict
        problem = json.loads(open(load_path).read())

        # read problem description
        self.label = problem['label']
        self.problem_type = problem['problem_type']
        self.model_label = problem['model_label']

        # load data
        self.data = pd.read_json(data_path)

    def create_model(self, num_iter, alpha):
        """
        Create models based on label value
        :param num_iter: number of iterations
        :param alpha: learning rate
        :return:
        """
        if self.model_label == 'NaiveBayes':
            # Create Naive Bayes
            self.model = NaiveBayes(self.data, self.label)

        elif self.model_label == "NeuralNetwork":
            # Create ANN
            self.model = NeuralNetwork(self.data, self.label, the_num_iter=num_iter)

        elif self.model_label == "SVM":
            # Create SVM
            self.model = SVM(self.data, self.label, the_num_iter=num_iter, the_alpha=alpha)

    def set_testing(self, file_path):
        """
        Set testing data
        :param file_path: path of testing data
        :return:
        """
        self.testing_data = pd.read_csv(file_path)

    def predict(self):
        """
        Predict on testing data
        :return:
        """
        self.model.predict(self.testing_data)

