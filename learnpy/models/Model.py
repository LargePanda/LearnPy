__author__ = 'Jiarui Xu'

from abc import ABCMeta, abstractmethod

# to be implemented next week

class Model(metaclass=ABCMeta):

    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def predict(self, data):
        pass

    @abstractmethod
    def summary(self):
        pass

    @abstractmethod
    def report(self):
        pass