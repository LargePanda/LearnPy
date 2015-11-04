__author__ = 'Jiarui Xu'
import pandas as pd


class Problem:

    def __init__(self, problem_type=None, file_path=None):

        # define problem type
        self.problem_type = problem_type
        self.check_type()

        # define data set
        self.file_path = file_path
        self.data = None
        self.read_data()

        # define label
        self.label = None
        self.check_label()


    def check_type(self):
        if self.problem_type in ['BinaryClassification', 'MultiClassification']:
            raise Exception('Problem Type Not Included.')

    def read_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
        except IOError:
            raise

    def set_label(self, label):
        if label in self.data.columns:
            self.label = label
            self.data[label] = self.data[label].astype(str)
        else:
            raise Exception("Label column does not exist.")

    def check_label(self):
        stat = pd.DataFrame.summary(self.data[self.label])
        if(self.problem_type == 'BinaryClassification' and stat['unique'] !=2) \
                or (self.problem_type == 'MultiClassification' and stat['unique'] <= 2):
            raise Exception("Number of unique items is not consistent with problem type")
