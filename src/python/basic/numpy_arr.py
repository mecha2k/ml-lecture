import numpy as np
from icecream import ic

X = np.arange(9)
X = X.reshape((3, 3)).astype(float)

xm1 = X.mean(axis=0)
xm2 = X.mean(axis=1)

ic(X)
ic(xm1)
ic(xm2)

xm1 = X - xm1
xm2 = X - xm2

ic(xm1)
ic(xm2)

y = np.arange(9)
y = y.reshape((3, 3))
ic(y)
y = y.reshape((3, 3, 1))
ic(y)

Value = np.array([[1, 2, 3, 4], [2, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]])
Value1 = np.array([1])
Value2 = np.array([3, 3, 3, 3])
Value3 = np.array([4, 5, 6, 7, 8]).reshape(5, 1)

ic(Value)
ic(Value1)
ic(Value2)
ic(Value3)
ic(Value + Value1)  # 4x4 + 1
ic(Value + Value2)  # 4x4 + 1x4
ic(Value2 + Value3)  # 1x4 + 4x1

weights = np.random.random(size=(5, 4))
ic(weights)
bb = np.sum(weights, axis=1)[:, np.newaxis]
ic(bb)
aa = weights / bb
ic(aa)

returns = np.array([1, 2, 3, 4, 5])
risks = np.random.random(size=5)
sharp_ratio = returns / risks
ic(returns)
ic(risks)
ic(sharp_ratio)
