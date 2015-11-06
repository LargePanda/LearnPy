__author__ = 'Jiarui Xu'

from sklearn import svm
import pandas as pd

data_path = "../../examples/data/iris.csv"

data = pd.read_csv(data_path)
clf = svm.SVC()
clf.fit(data.iloc[:, 0:4], data['Name'])

