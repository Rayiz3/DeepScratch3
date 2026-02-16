import numpy as np
import unittest

class Variable:
    def __init__(self, data):
        # exception flow
        if (data is not None) and (not isinstance(data, np.ndarray)):
            raise TypeError("type {} is not supported.".format(type(data)))
        
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        if self.grad is None: # if terminal variable (loss fuction value)
            self.grad = np.ones_like(self.data) # default grad = 1s (with same DT with self.data)
        
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)
            
            if x.creator is not None:
                funcs.append(x.creator)

class Function:
    def __call__(self, input):  # __call__ : called when f(...) (f is an instance of the class)
        x = input.data
        y = as_array(self.forward(x))  # prevented : numpy changes (0-dim ndarray) into (scalar) when it is computed
        output = Variable(y)
        output.set_creator(self)
        
        self.input = input
        self.output = output
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.input.data
        return 2 * x * gy

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        return np.exp(x) * gy

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)  # assertEqual : whether the values are equal
        
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)
        
    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg  = np.allclose(x.grad, num_grad)  # allclose(a, b, atol, rtol) : whether a and b are 'close' together
                                              # close only if : |a-b| <= (atol + rtol * |b|)
        self.assertTrue(flg)

# rapper functions
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

# diff function
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

# change (np.othertype) => (np.ndarray)
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

unittest.main()
