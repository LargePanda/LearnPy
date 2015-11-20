__author__ = 'Jiarui Xu'

from learnpy.Problem import Problem

# create a problem with training data
pro = Problem("BinaryClassification", "./data/iris_training.csv")

# set the predictor variable
pro.set_label('Name')

# save the problem
pro.save("problem.json", "data.json")
# load the problem
pro.load("problem.json", "data.json")

pro.set_model("SVM")
pro.model.fit(None)

# set testing data
pro.set_testing("./data/iris_testing.csv")
pro.predict()


# new problem for 242 demo

# create a problem
pro2 = Problem("BinaryClassification", "./data/lin_training.csv")

# set the predictor variable
pro2.set_label('Name')

pro2.set_model("SVM")
pro2.model.fit(None)

pro2.set_testing("./data/lin_testing.csv")
pro2.predict()

pro2.model.plot_demo()
