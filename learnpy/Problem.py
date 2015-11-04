__author__ = 'Jiarui Xu'
import pandas as pd


class Problem:

    def __init__(self, problem_type=None, file_path=None):

        # define problem type
        self.problem_type = type
        self.check_type()

        # define data set
        self.file_path = file_path
        self.data = None
        self.read_data()

        # define label
        self.label = None

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
        else:
            raise Exception("Label column does not exist.")