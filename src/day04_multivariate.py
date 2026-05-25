import numpy as np

##1.
data = np.loadtxt('data/houses.txt', delimiter=',')
X = data[: , :2]
y = data[: , 2]
"""print(X.shape)
print(y.shape)
print(X[:5])"""


###2.

def compute_cost_multi(X, y, w, b):
    m = X.shape[0]
    predictions = X @ w + b
    errors = predictions - y
    cost = (1/(2*m)) * np.dot(errors,errors)
    return cost

w_init = np.zeros(2)
b_init = 0
#print(compute_cost_multi(X, y, w_init, b_init)) 

###3.
   
def compute_gradient_multi(X, y, w, b):
    m = X.shape[0]
    predictions = X @ w + b        
    errors = predictions - y       
    
    dj_dw = (1 / m) * (X.T @ errors)  
    dj_db = (1 / m) * np.sum(errors)
    
    return dj_dw, dj_db

dj_dw, dj_db = compute_gradient_multi(X, y, w_init , b_init)
print(dj_dw)   
print(dj_db)   

###4.

def gradient_descent_multi(X, y, w_init, b_init, alpha, num_iters):
    w = w_init.copy()
    b = b_init
    history = []
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_multi(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost= compute_cost_multi(X,y,w,b)
        history.append(cost)
    return w, b, history


###5.


w_init = np.zeros(2)
b_init = 0
alpha = 1e-7
num_iters = 1000

w, b, history = gradient_descent_multi(X, y, w_init, b_init, alpha, num_iters)
print(f"w: {w}, b: {b:.2f}")
print(f"Final cost: {history[-1]:.2f}")


###6.

import matplotlib.pyplot as plt

plt.plot(history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Day 4 - Loss Curve')
plt.savefig('plots/day04_loss_curve.png')
plt.show()
