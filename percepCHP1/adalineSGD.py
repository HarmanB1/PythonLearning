class AdalineSGD:
    def __init__(self, eta=0.01, nIter=30, randomState=1):
        self.eta = eta
        self.nIter = nIter
        self.randomState = randomState

        """
        where X is the data [examples, features], y is the actual definition
        """
    def fit(self, X, y):
