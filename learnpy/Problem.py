__author__ = 'Jiarui Xu'
import pandas as pd


class Problem:

    def __init__(self, problem_type=None, file_path=None):

        # define problem type
        self.problem_type = type
        self.check_type()

        # define data set
        self.file_path = file_path
        self.read_data()

    def check_type(self):
        if self.problem_type in ['BinaryClassification', 'MultiClassification']:
            raise Exception('Problem Type Not Included.')

    def read_data(self):
        try:
            self.data = open(self.file_path, 'r')
        except IOError:
            raise