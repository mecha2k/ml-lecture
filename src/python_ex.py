a = list(map(lambda x: x + 2, range(5)))
print(a)

from functools import reduce

b = reduce(lambda x, y: x + y, range(5))
print(b)

c = list(filter(lambda x: x < 5, range(10)))
print(c)

A = [[1, 2, 3], [1, 2, 3]]
B = [[4, 5, 6], [4, 5, 6]]

extend_a = []
extend_a.extend(A)
extend_a.extend(B)
print(extend_a)

append_a = []
append_a.append(A)
append_a.append(B)
print(append_a)

import numpy as np

a = np.random.choice(10, 5)
print(a)

a = np.random.choice(5, 3, p=[0.5, 0.2, 0.1, 0.1, 0.1])
print(a)