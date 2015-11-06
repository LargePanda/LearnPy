__author__ = 'Jiarui Xu'

import unittest
from learnpy.Problem import Problem


class ProblemTest(unittest.TestCase):
    def setUp(self):
        self.problem = Problem("BinaryClassification", "../../../examples/data/iris.csv")
        self.problem.set_label("Name")

    def tearDown(self):
        self.problem = None

    def test_data_read(self):
        self.assertEqual(len(self.problem.data), 100)

    def test_problem_reload(self):
        self.problem.save("test_problem.json", "test_data.json")
        self.problem.load("test_problem.json", "test_data.json")
        self.test_problem_descriptions()

    def test_problem_descriptions(self):
        self.assertEqual(self.problem.label, "Name")
        self.assertEqual(self.problem.problem_type, "BinaryClassification")
        self.assertEqual(self.problem.file_path, "../../../examples/data/iris.csv")


suite = unittest.TestLoader().loadTestsFromTestCase(ProblemTest)
unittest.TextTestRunner().run(suite)