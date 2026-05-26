"""
Day 6: Polynomial Features
- Ne yapar: sentetik veri üretip lineer vs polinom model karşılaştırır
- Denklem: f(x) = w1*x + w2*x^2 + w3*x^3 + b
- Veri: sentetik, x = 0..19, y = x^2 + noise
"""
###1.

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

x = np.arange(0, 20, 1)
y = x**2 + np.random.randn(20) * 5

plt.scatter(x,y)
plt.ylabel("y-axis")
plt.xlabel("x-axis")
plt.title("Scatter Example")
plt.savefig("plots/day06_scatter_example.png")
plt.show


###2.

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from day05_feature_scaling import zscore_normalize
from day03_gradient_descent import gradient_descent

x_norm, mu, sigma = zscore_normalize(x)
w, b, history = gradient_descent(x_norm,y,0,0,1,1000)

print("Test: " , history[-1])

y_pred = w * x_norm + b

plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.title("Linear Fit")
plt.show()



###3.


from day04_multivariate import gradient_descent_multi

X_poly = np.c_[x, x**2, x**3 ]
X_poly_norm, mu, sigma = zscore_normalize(X_poly)


###4.


w, b, history = gradient_descent_multi(X_poly_norm, y, np.zeros(X_poly.shape[1]), 0, 0.1, 1000)
print(w, b)

y_pred_poly = X_poly_norm @ w + b

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(x, y)
ax1.plot(x, y_pred, color='red')
ax1.set_title("Linear Fit")

ax2.scatter(x, y)
ax2.plot(x, y_pred_poly, color='orange')
ax2.set_title("Polynomial Fit")
plt.savefig("plots/day06_polynomial_fit.png")
plt.show()





