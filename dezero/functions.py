import numpy as np
from dezero.core import Function
from dezero.core import as_variable


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
    
class Tanh(Function):
    def forward(self, x):
        return np.tanh(x)
    
    def backward(self, gy):
        x, = self.inputs
        return gy * (1 - tanh(x) ** 2)

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        return np.reshape(x, self.shape)  # numpy function

    def backward(self, gy):
        return reshape(gy, self.x_shape)  # Dezero function (gy is Variable, not ndarray)

class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes
    
    def forward(self, x):
        return np.Transpose(self.axes)
    
    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)
        
        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))  # argsort : list of index that makes the list sequential(increase)
        return transpose(gy, inv_axes)


def sin(x):
    return Sin()(x)

def cos(x):
    return Cos()(x)

def tanh(x):
    return Tanh()(x)

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

def transpose(x, axes=None):
    return Transpose(axes)(x)
