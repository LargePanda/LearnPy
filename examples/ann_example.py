__author__ = 'Jiarui Xu'

from learnpy.Problem import Problem

# create a problem
pro = Problem("BinaryClassification", "./data/iris.csv")

# set the predictor variable
pro.set_label('Name')

# use the model neural network
pro.set_model("NeuralNetwork")

# fit the model into the data set above
pro.model.fit(None)