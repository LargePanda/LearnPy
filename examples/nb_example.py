__author__ = 'Jiarui Xu'

from learnpy.Problem import Problem

# create a problem
pro = Problem("BinaryClassification", "./data/iris.csv")

# set the predictor variable
pro.set_label('Name')

pro.set_model("NaiveBayes")
