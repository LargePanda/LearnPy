__author__ = 'Jiarui Xu'

import unittest
from learnpy.Problem import Problem
from learnpy.models.ModelUtil import *


class SVMTest(unittest.TestCase):
    def setUp(self):
        self.problem = Problem("BinaryClassification", "../../../examples/data/lin_training.csv")
        self.problem.set_label("Name")
        self.problem.set_model("SVM")
        self.data = self.problem.data
        self.model = self.problem.model

    def tearDown(self):
        self.problem = None
        self.data = None

    def test_fit_stats(self):
        self.model.fit(None)
        acc = calculate_acc(self.model.fit_report['output'], self.model.fit_report['target'])
        self.assertEqual(0.97668, round(acc, 5))


suite = unittest.TestLoader().loadTestsFromTestCase(SVMTest)
unittest.TextTestRunner().run(suite)