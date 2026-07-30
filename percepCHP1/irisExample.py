import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from perceptron import perceptron

link = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
print("From URL;", link)
dataFrame = pd.read_csv(link, header=None, encoding="utf-8")

print(dataFrame.tail())

y = dataFrame.iloc
