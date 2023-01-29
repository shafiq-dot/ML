import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12.0, 9.0)

dataset = pd.read_csv("diabetes.csv")

print(dataset.head())

x = dataset.iloc[:, 0]

y = dataset.iloc[:, 1]

plt.scatter(x, y)

plt.show()

x_mean = np.mean(x)

y_mean = np.mean(y)

num = 0

den = 0

for i in range(len(x)):
    num += (x[i] - x_mean) * (y[i] - y_mean)

    den += (x[i] - x_mean) ** 2

m = num / den

c = y_mean - m * x_mean

print(m, c)

y_pred = m * x + c

plt.scatter(x, y)

plt.plot([min(x), max(x)], [min(y_pred), max(y_pred)], color="red")

plt.show()