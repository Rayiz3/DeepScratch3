if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # to find 'dezero' directory wherever execute a .py file

import numpy as np
import math
from dezero import Function
from dezero import Variable


def rosenbrock(x0, x1):
    return 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2


x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.001
iters = 1000

# gradient descent
for i in range(iters):
    print(x0, x1)
    
    y = rosenbrock(x0, x1)
    x0.cleargrad()
    x1.cleargrad()
    y.backward()
    
    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad