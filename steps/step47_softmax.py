if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # to find 'dezero' directory wherever execute a .py file

import numpy as np
import dezero.functions as F
from dezero import Variable, as_variable
from dezero.models import MLP


def softmax1d(x):
    y = F.exp(as_variable(x))
    sum_y = F.sum(y)
    return y / sum_y

# get_item function
a = Variable(np.array([[1,2,3], [4,5,6]]))
b = a[:,2]
print(b)  # [3 6]

# softmax cross entropy
model = MLP((10, 3))

x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])
y = model(x)
loss = F.softmax_cross_entropy(y, t)
print(loss)