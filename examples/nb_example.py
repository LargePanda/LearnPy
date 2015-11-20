__author__ = 'Jiarui Xu'

from learnpy.Problem import Problem

# create a problem
pro = Problem("BinaryClassification", "./data/iris_training.csv")

# set the predictor variable
pro.set_label('Name')

pro.set_model("NaiveBayes")
pro.model.fit(None)

pro.set_testing("./data/iris_testing.csv")
pro.predict()


# new problem

# create a problem
pro2 = Problem("BinaryClassification", "./data/lin_training.csv")

# set the predictor variable
pro2.set_label('Name')

pro2.set_model("NaiveBayes")
pro2.model.fit(None)

pro2.set_testing("./data/lin_testing.csv")
pro2.predict()