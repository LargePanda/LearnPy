__author__ = 'Jiarui Xu'

__author__ = 'Jiarui Xu'

import unittest
from learnpy.Problem import Problem
from learnpy.models.ModelUtil import *

class ModelUtilTest(unittest.TestCase):
    def setUp(self):
        self.problem = Problem("BinaryClassification", "../../../examples/data/iris.csv")
        self.problem.set_label("Name")
        self.problem.set_model("NaiveBayes")
        self.data = self.problem.data

    def tearDown(self):
        self.problem = None
        self.data = None

    def test_unique_list(self):
        ulist = unique_list(self.data['Name'])
        self.assertEqual(set(['Iris-versicolor', 'Iris-setosa']), set(ulist))

    def test_get_both_columns(self):
        num_cols, cat_cols = get_both_columns(self.data, 'Name')
        self.assertEqual(set(['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']), set(num_cols))
        self.assertEqual(set(['Dum']), set(cat_cols))

    def test_get_mean_std(self):
        mean, std = get_mean_std(self.data['SepalWidth'])
        self.assertEqual(3.0940, round(mean, 4))
        self.assertEqual(0.4761, round(std, 4))

    def test_calculate_prob(self):
        prob = calculate_prob(10, 2, 3)
        self.assertEqual(0.0038, round(prob, 4))

    def test_nonlin(self):
        out = nonlin(0.5, False)
        self.assertEqual(0.62246, round(out, 5))
        out = nonlin(0.5, True)
        self.assertEqual(0.25, round(out, 5))

    def test_get_class(self):
        x = get_class(0.3)
        self.assertEqual(0, x)

    def test_get_class_np(self):
        x = get_class_np(-0.3)
        self.assertEqual(-1, x)

suite = unittest.TestLoader().loadTestsFromTestCase(ModelUtilTest)
unittest.TextTestRunner().run(suite)