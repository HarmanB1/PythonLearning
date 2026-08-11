import numpy as np


class adaline:
    def __init__(self, eta=0.01, nIter=50, randomState=10):
        self.eta = eta
        self.nIter = nIter
        self.randomState = randomState

    def fit(
        self, X, y
    ):  # X array like , gonna be = >J [nexample, nfeatures], y is the target values
        rgen = np.random.RandomState(self.randomState)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0)
        self.losses_ = []

        for i in range(self.nIter):
            netInput = np.dot(X, self.w_) + self.b_
            output = self.activation(netInput)
            errors = y - output  # actual - calculated
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(np.dot(X, self.w_) + self.b_) >= 0.5, 1, 0)
