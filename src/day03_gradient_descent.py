""" Day 3 — Gradient Descent (Univariate)
```
dJ/dw = (1/m) * sum( (f(x_i) - y_i) * x_i )
dJ/db = (1/m) * sum(  f(x_i) - y_i )
w := w - α * dJ/dw
b := b - α * dJ/db
```
"""
import numpy as np
import matplotlib.pyplot as plt
###1.
 
def compute_gradient(x,y,w,b):
    f = w * x + b
    m = len(x)
    dw = (f-y) * x
    db = (f-y)
    dj_dw = np.sum(dw) / m
    dj_db = np.sum(db) / m
    return dj_dw, dj_db

x = np.array([1.0, 2.0])
y = np.array([300.0, 500.0])
w, b = 200, 100  

dj_dw, dj_db = compute_gradient(x, y, w, b)
print(dj_dw, dj_db)  

###2.

def compute_cost(x, y, w, b):
    total_cost=0.0   
    for i in range(len(x)):
        f_xi = w * x[i] + b
        cost = (f_xi-y[i])**2
        total_cost += cost
    total_cost = (total_cost) / (2*len(x))
    return total_cost


def gradient_descent(x, y, w_init, b_init, alpha, num_iters):
    w = w_init
    b = b_init
    history=[]
    for i in range(num_iters):
        dj_dw, dj_db  = compute_gradient(x,y,w,b) 
        w -= alpha * dj_dw
        b -= alpha * dj_db
        J = compute_cost(x,y,w,b)
        history.append((w,b,J))
    return w, b, history

###3.

x = np.array([1.0, 2.0])
y = np.array([300.0, 500.0])
w_final, b_final, history = gradient_descent(x, y, 0, 0, 0.01, 100000)
print(w_final, b_final) 


###4.

J_history = [h[2] for h in history]
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("Cost J")
plt.title("Loss Curve")
plt.savefig("plots/day03_loss_curve.png")
plt.show()

plt.plot(J_history[:100])
plt.xlabel("Iteration")
plt.ylabel("Cost J")
plt.title("Loss Curve")
plt.savefig("plots/day03_loss_curve_zoom.png")
plt.show()

###5.
alphas = [1, 0.1, 0.01, 0.001]

for alpha in alphas:
    w, b, history = gradient_descent(x, y, 0, 0, alpha, 1000)
    J_history = [h[2] for h in history]
    print(f"alpha={alpha}, final w={w:.2f}, b={b:.2f}, J={J_history[-1]:.2f}")
        








