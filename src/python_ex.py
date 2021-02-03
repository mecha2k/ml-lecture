a = list(map(lambda x: x + 2, range(5)))
print(a)

from functools import reduce

# apply a particular function passed in its argument to all of the list elements
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


from types import SimpleNamespace

pong_dict = {
    "env_name": "PongNoFrameskip-v4",
    "stop_reward": 18.0,
    "run_name": "pong",
    "replay_size": 100000,
    "replay_initial": 10000,
    "target_net_sync": 1000,
    "epsilon_frames": 10 ** 5,
    "epsilon_start": 1.0,
    "epsilon_final": 0.02,
    "learning_rate": 0.0001,
    "gamma": 0.99,
    "batch_size": 32,
}
print(*pong_dict)

nameSpace = SimpleNamespace(**pong_dict)
print(nameSpace)

pong_dict["gamma"] = 0.9
print(nameSpace)

# *tuple means
# "treat the elements of this iterable as positional arguments to this function call"
# **dict means
# "treat the key-value in the dictionary as additional named arguments to this function call"
def func_a(x, y):
    print(x, y)


dict_sample = {"x": 1, "y": 2}
tuple_sample = (3, 2)

func_a(**dict_sample)
func_a(*tuple_sample)
