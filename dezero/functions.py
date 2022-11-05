import numpy as np
from dezero.core import Function


class Sin(Function):
    def forward(self, x):
        return np.sin(x)
    
    def backward(self, gy):
        x, = self.inputs  # x, : [Variable()] => Variable()
        return gy * cos(x)
    
class Cos(Function):
    def forward(self, x):
        return np.cos(x)
    
    def backward(self, gy):
        x, = self.inputs
        return gy * -sin(x)

def sin(x):
    return Sin()(x)

def cos(x):
    return Cos()(x)


