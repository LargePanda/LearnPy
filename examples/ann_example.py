__author__ = 'Jiarui Xu'

from learnpy.Problem import Problem

# create a problem
pro = Problem("BinaryClassification", "./data/iris_training.csv")

# set the predictor variable
pro.set_label('Name')

# use the model neural network
pro.set_model("NeuralNetwork", num_iter=1000)

# fit the model into the data set above
pro.model.fit(None)

# set testing data
pro.set_testing("./data/iris_testing.csv")
pro.predict()