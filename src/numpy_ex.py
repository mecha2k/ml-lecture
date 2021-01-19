import numpy as np

ax = np.arange(0, 20, 2)
ay = np.array([0, 1, 2, 2, 0, 1, 1, 1, 2, 0])
print(ax, ay)

axx = ax[(ay == 1) | (ay == 2)]
print(axx)

A = np.array([[1, 1], [0, 1]])
B = np.array([[2, 0], [3, 4]])
C = A @ B
D = A.dot(B.T)

print(C)
print(D)