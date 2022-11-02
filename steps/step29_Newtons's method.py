if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # to find 'dezero' directory wherever execute a .py file

import numpy as np
from dezero import Variable


def f(x):
    return x ** 4 - 2 * x ** 2

def gx2(x):
    return 12 * x ** 2 - 4

x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)
    
    y = f(x)
    x.cleargrad()
    y.backward()
    
    x.data -= x.grad / gx2(x.data)