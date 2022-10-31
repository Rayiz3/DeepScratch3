if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # to find 'dezero' directory wherever execute a .py file

import numpy as np
from dezero import Variable

def sphere(x, y):
    return x ** 2 + y ** 2

def matyas(x, y):
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

def goldstein(x, y):
    x2 = x ** 2
    y2 = y ** 2
    xy = x * y
    return (1 + (x + y + 1)**2 * (19 - 14*x + 3*x2 - 14*y + 6*xy + 3*y2)) * \
           (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x2 + 48*y - 36*xy + 27*y2))

x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = sphere(x, y)
z.backward()
print(x.grad, y.grad)

a = Variable(np.array(1.0))
b = Variable(np.array(1.0))
c = matyas(a, b)
c.backward()
print(a.grad, b.grad)

p = Variable(np.array(1.0))
q = Variable(np.array(1.0))
r = goldstein(p, q)
r.backward()
print(p.grad, q.grad)