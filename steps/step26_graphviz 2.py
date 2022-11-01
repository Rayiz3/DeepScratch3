if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # to find 'dezero' directory wherever execute a .py file

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph

def goldstein(x, y):
    x2 = x ** 2
    y2 = y ** 2
    xy = x * y
    return (1 + (x + y + 1)**2 * (19 - 14*x + 3*x2 - 14*y + 6*xy + 3*y2)) * \
           (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x2 + 48*y - 36*xy + 27*y2))

p = Variable(np.array(1.0))
q = Variable(np.array(1.0))
r = goldstein(p, q)
r.backward()

p.name = 'p'
q.name = 'q'
r.name = 'r'
plot_dot_graph(r, verbose=False, to_file='goldstein.png')