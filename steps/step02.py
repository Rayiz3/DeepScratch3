import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data

class Function:
    def __call__(self, input):  # __call__ : called when f(...) (f is an instance of the class)
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    
    def forward(self, x):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2

x = Variable(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)