__author__ = 'Jiarui Xu'

import unittest
from learnpy.Problem import Problem
from learnpy.models.ModelUtil import *


class NeuralNetworkTest(unittest.TestCase):
    def setUp(self):
        self.problem = Problem("BinaryClassification", "../../../examples/data/iris.csv")
        self.problem.set_label("Name")
        self.problem.set_model("NeuralNetwork")
        self.data = self.problem.data
        self.model = self.problem.model

    def tearDown(self):
        self.problem = None
        self.data = None

    def test_fit_stats(self):
        self.model.fit()
        acc = calculate_acc(self.model.fit_report['output'], self.model.fit_report['target'])
        self.assertEqual(1.0, acc)

    def test_fit_stats_oneIter(self):
        self.model.num_iter = 1
        self.model.fit()
        acc = calculate_acc(self.model.fit_report['output'], self.model.fit_report['target'])
        self.assertEqual(.5, acc)

suite = unittest.TestLoader().loadTestsFromTestCase(NeuralNetworkTest)
unittest.TextTestRunner().run(suite)