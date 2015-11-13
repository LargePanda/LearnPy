__author__ = 'Jiarui Xu'

__author__ = 'Jiarui Xu'

import unittest
from learnpy.Problem import Problem
from learnpy.models.ModelUtil import *


class NaiveBayesTest(unittest.TestCase):
    def setUp(self):
        self.problem = Problem("BinaryClassification", "../../../examples/data/iris.csv")
        self.problem.set_label("Name")
        self.problem.set_model("NaiveBayes")
        self.data = self.problem.data
        self.model = self.problem.model

    def tearDown(self):
        self.problem = None
        self.data = None

    def test_prob_stats(self):
        prob = self.model.prob_hub['Dum']['Iris-versicolor']['a']
        self.assertEqual(0.40, round(prob, 2))

        prob = self.model.prob_hub['Dum']['Iris-setosa']['b']
        self.assertEqual(0.26, round(prob, 2))

    def test_mean_std(self):
        mean, std = get_mean_std(self.data['SepalWidth'])
        self.assertEqual(3.0940, round(mean, 4))
        self.assertEqual(0.4761, round(std, 4))


suite = unittest.TestLoader().loadTestsFromTestCase(NaiveBayesTest)
unittest.TextTestRunner().run(suite)