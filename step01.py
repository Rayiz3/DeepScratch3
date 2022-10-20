# https://github.com/WegraLee/deep-learning-from-scratch-3

import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data


data = np.array(1.0)
x = Variable(data)
print(x.data)

x.data = np.array(2.0)
print(x.data)

y = np.array([[1,2,3], [4,5,6]])