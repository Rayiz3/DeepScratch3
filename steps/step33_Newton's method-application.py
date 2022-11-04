if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # to find 'dezero' directory wherever execute a .py file

import numpy as np
from dezero import Variable


def f(x):
    return x ** 4 - 2 * x ** 2

x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)
    
    y = f(x)
    x.cleargrad()
    y.backward(create_graph=True)
    
    gx = x.grad
    x.cleargrad()  # prevented : x.grad value is accumulated; f'(1)() + f'(2)() + ...
    gx.backward()
    gx2 = x.grad
    
    x.data -= gx.data / gx2.data