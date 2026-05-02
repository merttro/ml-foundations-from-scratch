"""
Day 1 — NumPy Warmup
Egzersizler: vektör/matris işlemleri, dot product, houses.txt yükleme
Veri: data/houses.txt
"""

import numpy as np
x = np.array([1.0,2.0,3.0,4.0])
y = np.array([2.0,4.0,6.0,8.0])
print("x shape:", x.shape) ## (4,) ==> 4 elemanlı bir boyutlu vektör
print("y shape:", y.shape) ## (4,) ==> 4 elemanlı bir boyutlu vektör

##########

def f_wb (a):
    w=2
    b=0
    output=np.dot(w,a) + b
    return output
print(f_wb(x))

#######

X = np.array([[1,2],
              [3,4],
              [5,6]])

XT = X.T ### Matris Transpozu
print("X.T @ X shape:", (XT @ X).shape) ## sonucların farklı olmasının sebebi matris çarpımının kurallarından kaynaklıdır
print("X @ X.T shape:", (X @ XT).shape)
#######

print(np.dot(x,y))
sum=0.0
for i in range(len(x)):
    sum_i= x[i] * y[i]
    sum += sum_i
print(sum)
print(np.allclose(np.dot(x,y),sum)) ##`np.dot(x, y)` ile manuel `sum(x[i]*y[i] for i in range(len(x)))` aynı sonucu mu veriyor? Karşılaştır.

########
data = np.loadtxt("data/houses.txt",delimiter=",")
print("Shape: " , data.shape)
print("İlk 5 satır: " , data[:5])



