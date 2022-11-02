if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # to find 'dezero' directory wherever execute a .py file

import numpy as np
import math
from dezero import Function
from dezero import Variable


# taylor series approximation
def my_sin(x, threshold=0.0001):
    y = 0 
    for i in range(100000):
        c = (-1)**i * x**(2 * i + 1) / math.factorial(2 * i + 1)
        y = y + c
        if abs(c.data) < threshold:
            break
    return y
