import numpy as np


class AdalineCustom:
    def __init__(self, eta=0.3, epoch=30, randomState=1):
        self.eta = eta
        self.epoch = epoch
        self.randomState = randomState

    # X , is [examples, features]

    def initWeights(self, X):
        rng = np.random.default_rng(self.randomState)
        self.w_ = rng.normal(loc=0.0, scale=0.1, size=X.shape[1])
        self.b_ = np.float_(0.0)

    def calc(self, X):
        return np.dot(X, self.w_) + self.b_

    def activation(self, score):
        return score

    def fit(self, X, y):
        self.initWeights(X)
        for _ in range(self.epoch):
            output = self.calc(X)
            output = self.activation(output)
            error = y - output
            self.w_ += (
                self.eta * 2 * X.T.dot(error) / X.shape[0]
            )  # take all x1 features, mult by error for their error, add it and dvide by how many features to get avg
            self.b_ += self.eta * 2 * error.mean()

    def predict(self, X):
        scores = self.calc(X)
        scores = self.activation(scores)

        return np.where(scores >= 0.0, 1, 0)
