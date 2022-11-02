if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # to find 'dezero' directory wherever execute a .py file

import numpy as np
import math
from dezero import Function
from dezero import Variable

class Sin(Function):
    def forward(self, x):
        return np.sin(x)
    
    def backward(self, gy):
        x = self.inputs[0].data
        return gy * np.cos(x)


# rapper function
def sin(x):
    return Sin()(x)

# taylor series approximation
def my_sin(x, threshold=0.0001):
    y = 0 
    for i in range(100000):
        c = (-1)**i * x**(2 * i + 1) / math.factorial(2 * i + 1)
        y = y + c
        if abs(c.data) < threshold:
            break
    return y


x = Variable(np.array(np.pi/4))
y = my_sin(x)
y.backward()

print(y.data)
print(x.data)