from adalineSGD import AdalineSGD as ada
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

s = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

df = pd.read_csv(s, header=None, encoding="utf-8")

y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", 0, 1)

X = df.iloc[0:100, [0, 2]].values

# standardize
X_std = np.copy(X)

# standairze first fieature = feature - mean / std
X_std[:, 0] = X[:, 0] - X[:, 0].mean() / X[:, 0].std()


X_std[:, 1] = X[:, 1] - X[:, 1].mean() / X[:, 1].std()

adaline = ada()
adaline.fit(X_std, y)

plt.figure(figsize=(8, 6))

plt.plot(range(1, len(adaline.losses_) + 1), adaline.losses_, marker="o", color="b")

plt.xlabel("epochs")
plt.ylabel("error")
plt.show()
