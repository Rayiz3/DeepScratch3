import numpy as np


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
    def __call__(self, *inputs):  # __call__ : called when f(...) (f is an instance of the class)
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # unpacking : f([x0, x1]) => f(x0, x1)
        if not isinstance(ys, tuple): ys = (ys,)  # if forward result is not a tuple
        outputs = [Variable(as_array(y)) for y in ys]  # prevented : numpy changes (0-dim ndarray) into (scalar) when it is computed
        
        for output in outputs:
            output.set_creator(self)
        
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]  # if result is (y,), just return y
    
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

class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1

# rapper functions
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def add(x0, x1):
    return Add()(x0, x1)

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


# add : forward
x0 = Variable(np.array(2))
x1 = Variable(np.array(3))
y = add(x0, x1)
print(y.data)