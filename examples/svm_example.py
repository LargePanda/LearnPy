__author__ = 'Jiarui Xu'

from learnpy.Problem import Problem

# create a problem
pro = Problem("BinaryClassification", "./data/iris.csv")

# set the predictor variable
pro.set_label('Name')

# save the problem
pro.save("problem.json", "data.json")
# load the problem
pro.load("problem.json", "data.json")
