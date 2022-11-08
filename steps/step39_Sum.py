if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # to find 'dezero' directory wherever execute a .py file

import numpy as np
from dezero import Variable
import dezero.functions as F


x = Variable(np.array([[1,2,3], [4,5,6]]))
y = F.sum(x, axis=0)
y.backward()
print(y)
print(x.grad)

x = Variable(np.random.randn(2,3,4,5))
y = x.sum(keepdims=True)
print(y.shape)