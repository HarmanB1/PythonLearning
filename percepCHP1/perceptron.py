import numpy as np


class perceptron:
    def __init__(self, eta=0.01, nIter=50, randomState=1):
        self.eta = eta
        self.nIter = nIter
        self.randomState = randomState

    def fit(self, X, y):
        rgen = np.random.RandomState(self.randomState)
        self.w_ = rgen.normal(
            loc=0.0, scale=0.01, size=X.shape[1]
        )  # guassian vec -> weight genrator
        self.b_ = np.float_(0.0)
        self.errors_ = []

        for _ in range(self.nIter):
            errors = 0
            for xi, target in zip(X, y):  # pairs exmaple with real out
                predictY = np.dot(xi, self.w_) + self.b_
                deltaW = self.eta * (target - predictY) * xi
                deltab = self.eta * (target - predictY)

                self.w_ += deltaW
                self.b_ += deltab

                errors += int((target - predictY) != 0.0)
                self.errors_.append(errors)
        return self

    def predict(self, X):
        return np.where((np.dot(X, self.w_) + self.b_) >= 0.0, 1, 0)
