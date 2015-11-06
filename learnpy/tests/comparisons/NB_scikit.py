__author__ = 'Jiarui Xu'

from sklearn.naive_bayes import GaussianNB

import pandas as pd

data_path = "../../examples/data/iris.csv"

data = pd.read_csv(data_path)
gnb = GaussianNB()

# add evaluation later
# y_pred = gnb.fit(data.iloc[:, 0:4], data['Name']).predict(data.iloc[:, 0:4])
