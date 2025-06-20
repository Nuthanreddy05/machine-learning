# models.py
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

class LDAModel:
    def __init__(self):
        self.LinearDiscriminantAnalysis = LinearDiscriminantAnalysis()
    def fit(self, X, y):
        return self.LinearDiscriminantAnalysis.fit(X, y)

    def predict(self, X):
        return self.LinearDiscriminantAnalysis.predict(X)

class QDAModel:
    def __init__(self):
        self.QuadraticDiscriminantAnalysis = QuadraticDiscriminantAnalysis()

    def fit(self, X, y):
        self.QuadraticDiscriminantAnalysis.fit(X, y)

    def predict(self, X):
        return self.QuadraticDiscriminantAnalysis.predict(X)

class GaussianNBModel:
    def __init__(self):
        self.GaussianNB = GaussianNB()

    def fit(self, X, y):
        self.GaussianNB.fit(X, y)

    def predict(self, X):
        return self.GaussianNB.predict(X)
