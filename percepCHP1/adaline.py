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

        for i in range(self.nIter):
