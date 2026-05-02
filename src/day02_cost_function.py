"""
Day 2: Cost Function (Univariate Linear Regression)
Equation: J(w,b) = (1/2m) * sum((w*x_i + b - y_i)^2)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


### 1.
def compute_cost(x, y, w, b):
    total_cost = 0.0
    for i in range(len(x)):
        f_xi = w * x[i] + b
        cost = (f_xi - y[i]) ** 2
        total_cost += cost
    total_cost = (total_cost) / (2 * len(x))
    return total_cost


### 2.
def compute_cost_vectorized(x, y, w, b):
    f = w * x + b
    cost = (f - y) ** 2
    return np.sum(cost) / (2 * len(x))


### 3.
x = np.array([1.0, 2.0])
y = np.array([300.0, 500.0])

j1 = compute_cost(x, y, 200, 100)
j2 = compute_cost_vectorized(x, y, 200, 100)
print(np.allclose(j1, j2))
print(j1, j2)

### 4.
w_values = np.linspace(0, 400, 100)
cost_values = []
for w in w_values:
    cost_values.append(compute_cost(x, y, w, 100))
plt.plot(w_values, cost_values)
plt.xlabel("w")
plt.ylabel("J(w, b)")
plt.title("Cost vs w")
Path("plots").mkdir(exist_ok=True)
plt.savefig("plots/day02_cost_vs_w.png")
plt.show()

### 5.
x_big = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y_big = np.array([250, 300, 480, 430, 630, 730])

j = compute_cost(x_big, y_big, 209, 2.4)
print(j)
