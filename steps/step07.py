import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None
    
    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        f = self.creator  # 1. call creator func
        if f is not None:
            x = f.input  # 2. call last variable
            x.grad = f.backward(self.grad)  # 3. backward computation
            x.backward()  # 4. recursive, until meet user-defined variable

class Function:
    def __call__(self, input):  # __call__ : called when f(...) (f is an instance of the class)
        x = input.data
        y = self.forward(x)
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


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


A = Square()
B = Exp()
C = Square()

# forward
x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

assert y.creator == C  # assert .. : make exception if the following statement is false
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x

y.grad = np.array(1.0)
y.backward()
print(x.grad)