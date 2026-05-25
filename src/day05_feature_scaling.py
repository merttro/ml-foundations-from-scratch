"""
Day 5 — Feature Scaling + Learning Rate
Z-score normalization ile multivariate gradient descent'i hızlandırmak.
Veri: data/houses.txt (size, bedrooms → price)
Denklem: x_scaled = (x - mu) / sigma
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from day04_multivariate import compute_gradient_multi, compute_cost_multi

def zscore_normalize(X):
    mu = np.mean(X, axis=0)      # her sütunun ortalaması
    sigma = np.std(X, axis=0)    # her sütunun standart sapması
    X_norm = (X - mu) / sigma    # formül: (x - ortalama) / std
    return X_norm, mu, sigma

if __name__ == "__main__":
    # Task 1 test
    X_test = np.array([[1.0, 2.0],
                        [3.0, 4.0],
                        [5.0, 6.0]])
    X_norm, mu, sigma = zscore_normalize(X_test)
    print("Normalized means (0'a yakın olmalı):", np.mean(X_norm, axis=0))
    print("Normalized stds  (1'e yakın olmalı):", np.std(X_norm, axis=0))

    ###2.

    data = np.loadtxt('data/houses.txt', delimiter=',')
    X = data[:, :2]
    y = data[:, 2]

    X_norm, mu, sigma = zscore_normalize(X)

    print("\n--- House Verisi Normalizasyon ---")
    print(f"mu    (ortalama): {mu}")
    print(f"sigma (std):      {sigma}")
    print(f"X_norm min: {X_norm.min(axis=0)}")
    print(f"X_norm max: {X_norm.max(axis=0)}")

###3.
from day04_multivariate import gradient_descent_multi
w_init = np.zeros(X_norm.shape[1])
b_init = 0
alpha = 0.1
num_iters = 1000
w, b ,history = gradient_descent_multi(X_norm, y, w_init, b_init, alpha, num_iters)
print(f"Final w: {w}")
print(f"Final b: {b:.2f}")
print(f"Final cost: {history[-1]:.2f}")

###4.
w_init = np.zeros(X_norm.shape[1])
b_init = 0
alpha = [0.001 , 0.01 , 0.1 , 1.0]
num_iters = 1000
plt.figure()
for a in alpha:
    _, _, hist = gradient_descent_multi(X_norm, y, w_init, b_init, a, num_iters)
    plt.plot(hist, label="alpha=" + str(a))

plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Day 5 - Alpha Comparison')
plt.legend()
plt.savefig('plots/day05_alpha_compare.png')
plt.show()


###5.

x_new = [1200,3]
x_new_norm = (x_new - mu) / sigma
predict = np.dot(x_new_norm, w) + b
prediction = np.dot(x_new_norm, w) + b
print(f"\nTahmin (1200 sqft, 3 yatak): ${prediction:,.0f}")


