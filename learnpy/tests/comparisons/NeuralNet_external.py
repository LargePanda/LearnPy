__author__ = 'Jiarui Xu'

from NeuralNetworkExternal import NeuralNetwork
import numpy as np
import pandas as pd


data_path = "../../examples/data/iris.csv"

data = pd.read_csv(data_path)

# 1 is Iris-setosa and 0 is Iris-versicolor
# data['Name'] = data.Name.apply(lambda x: 1 if x == "Iris-setosa" else 0)


nn = NeuralNetwork.NeuralNetwork([2,2,1])

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 1, 1, 0])

nn.fit(X, y)

for e in X:
    print(e,nn.predict(e))
