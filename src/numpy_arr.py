import numpy as np

X = np.arange(9)
X = X.reshape((3, 3)).astype(float)

xm1 = X.mean(axis=0)
xm2 = X.mean(axis=1)

print(X)
print(xm1)
print(xm2)

xm1 = X - xm1
xm2 = X - xm2

print(xm1)
print(xm2)

y = np.arange(9)
y = y.reshape((3, 3))
print(y)
y = y.reshape((3, 3, 1))
print(y)
