import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from perceptron import perceptron

link = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
print("From URL;", link)
dataFrame = pd.read_csv(link, header=None, encoding="utf-8")

print(dataFrame.tail())

y = dataFrame.iloc[0:100, 4].values  # values converts to numpy arr
y = np.where(y == "Iris-setosa", 0, 1)
X = dataFrame.iloc[0:100, [0, 2]].values  # gets first and 3rd cols

plt.scatter(X[:50, 0], X[:50, 1], color="red", marker="o", label="Setosa")
plt.scatter(X[50:100, 0], X[50:100, 1], color="blue", marker="s")


ppn = perceptron()
ppn.fit(X, y)
print(ppn.predict(X))
