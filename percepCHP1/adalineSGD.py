import numpy as np


class AdalineSGD:
    def __init__(self, eta=0.01, nIter=30, randomState=1):
        self.eta = eta
        self.nIter = nIter
        self.randomState = randomState

        """
        where X is the data [examples, features], y is the actual definition
        """

    def _init_weights(self, m):
        self.rgen = np.random.RandomState(self.randomState)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float_(0.0)

    def activation(self, X):
        return X

    def _update_weights(self, xi, target):
        out = self.activation(np.dot(xi, self.w_) + self.b_)
        error = target - out
        self.w_ += self.eta * 2.0 * xi * (error)
        self.b_ += self.eta * 2.0 * error
        loss = error**2
        return loss

    def fit(self, X, y):
        self._init_weights(X.shape[0])
        self.losses_ = []

        losses = []
        for i in range(self.nIter):
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
                avgLoss = np.mean(losses)
                self.losses_.append(avgLoss)

    def predict(self, X):
        return np.where(self.activation(np.dot(X, self.w_) + self.b_) >= 0.5, 1, 0)
