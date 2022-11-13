import numpy as np
import dezero
from dezero import utils
from dezero.core import Function, as_variable


# =====================
# Basic function
# =====================
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

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        return gy * y

class Log(Function):
    def forward(self, x):
        y = np.log(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        return gy / x

class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        y = np.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        return gy * mask

# =====================
# Matrix topology
# =====================
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
        return np.transpose(x, self.axes)
    
    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)
        
        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))  # argsort : list of index that makes the list sequential(increase)
        return transpose(gy, inv_axes)

class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        self.x_shape = x.shape
        y = x[self.slices]
        return y

    def backward(self, gy):
        return get_item_grad(gy, self.slices, self.x_shape)
    
class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, x):
        y = np.zeros(self.in_shape)
        np.add.at(y, self.slices, x)  # add.at : add 'slices - index of x' to y
        return y

    def backward(self, gy):
        return get_item(gy, self.slices)

# =====================
# Matrix operation
# =====================
class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis  # index where going to sum  e.g.) axis=1 : sum(x[][i][]...)
        self.keepdims = keepdims  # if keep dimensional shape of output  e.g.) keepdims=True : return [[21]]
        
    def forward(self, x):
        self.x_shape = x.shape
        return np.sum(x, axis=self.axis, keepdims=self.keepdims)
    
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        return broadcast_to(gy, self.x_shape)

class Max(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        y = x.max(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        y = self.outputs[0]()  # weakref

        shape = utils.max_backward_shape(x, self.axis)  # e.g.) axis = 1 : [[a], [b], [c]]
        gy = reshape(gy, shape)
        y = reshape(y, shape)
        cond = (x.data == y.data)  # x: [[1, 2, 3], [4, 5, 6], [7, 8, 9]] <=> y: [[3], [6], [9]]
        gy = broadcast_to(gy, cond.shape)
        return gy * cond
    
class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        return np.broadcast_to(x, self.shape)

    def backward(self, gy):
        return sum_to(gy, self.x_shape)

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        return utils.sum_to(x, self.shape)
    
    def backward(self, gy):
        return broadcast_to(gy, self.x_shape)

class Matmul(Function):
    def forward(self, x, W):
        return x.dot(W)
    
    def backward(self, gy):
        x, W = self.inputs
        return matmul(gy, W.T), matmul(x.T, gy)

class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb

# =====================
# Loss function
# =====================
class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        return (diff ** 2).sum() / len(diff)
    
    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx = gy * (2. / len(diff)) * diff
        return gx, -gx

class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        N = x.shape[0]
        
        p = clip(softmax(x), 1e-15, 1.0)  # clip() : x range is fixed with (1e-15 ~ 1.0)
        logp = log(p)
        tlogp = logp[np.arange(N), t]
        y = -1/N * tlogp.sum()
        return y.data

    def backward(self, gy):
        x, t = self.inputs
        N, class_N = x.shape

        y = softmax(x)
        t_onehot = np.eye(class_N, dtype=t.dtype)[t.data]  # convert t to one-hot vector
        y = 1/N * gy * (y - t_onehot)
        return y

# =====================
# Activation function
# =====================
class Sigmoid(Function):
    def forward(self, x):
        # xp = cuda.get_array_module(x)
        # return xp.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation
        return 1 / (1 + np.exp(-x))

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx

class Softmax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        y = x - x.max(axis=self.axis, keepdims=True)  # prevented : overflow
        y = np.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)  # broadcast to each axis  e.g.) [[a], [b]] => [[0.2, 0.3, 0.5], [0.1, 0.3, 0.6]]
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx  # y * gy - y * sum(y * gy)

# rapper functions #
def sin(x):
    return Sin()(x)

def cos(x):
    return Cos()(x)

def tanh(x):
    return Tanh()(x)

def exp(x):
    return Exp()(x)

def log(x):
    return Log()(x)

def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)

def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)

def transpose(x, axes=None):
    return Transpose(axes)(x)

def get_item(x, slices):
    return GetItem(slices)(x)

def get_item_grad(x, slices, in_shape):
    return GetItemGrad(slices, in_shape)(x)

def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

def max(x, axis=None, keepdims=False):
    return Max(axis, keepdims)(x)

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

def matmul(x, W):
    return Matmul()(x, W)

def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)

def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)

def linear(x, W, b=None):
    return Linear()(x, W, b)

def sigmoid(x):
    return Sigmoid()(x)

def softmax(x, axis=1):
    return Softmax(axis)(x)