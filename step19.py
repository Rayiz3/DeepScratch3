from multiprocessing.sharedctypes import Value
import numpy as np
import weakref
import contextlib

class Config:  # for mode switching
    enable_backprop = True

@contextlib.contextmanager  # provide context : preprocessing ~ yield ~ postprocessing
def using_config(name, value):
    # preprocessing
    old_value = getattr(Config, name)  # old_value = Config.name
    setattr(Config, name, value)  # Config.name = value
    # preprocessing
    try:
        yield  # similar with return, but yield whenever it is called
    finally:
    # postprocessing
        setattr(Config, name, old_value)  # Config.name = old_value
    # postprocessing

class Variable:
    def __init__(self, data):
        # exception flow
        if (data is not None) and (not isinstance(data, np.ndarray)):
            raise TypeError("type {} is not supported.".format(type(data)))
        
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0
    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def backward(self, retain_grad=False):
        if self.grad is None: # if terminal variable (loss fuction value)
            self.grad = np.ones_like(self.data) # default grad = 1s (with same DT with self.data)
        
        funcs = []
        seen_set = set()
        
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)
                
        add_func(self.creator)
        
        while funcs:
            f = funcs.pop()
            xs, ys = f.inputs, [output() for output in f.outputs]
            gys = [y.grad for y in ys]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple): gxs  = (gxs,)
            
            for x, gx in zip(xs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx  # if using += : become *(x.grad) == *(y.grad)
                
                if x.creator is not None:
                    add_func(x.creator)
                    
            if not retain_grad: # remove unecessary grad variable
                for y in ys:
                    y.grad = None
    
    def cleargrad(self):
        self.grad = None
            

class Function:
    def __call__(self, *inputs):  # __call__ : called when f(...) (f is an instance of the class)
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # unpacking : f([x0, x1]) => f(x0, x1)
        if not isinstance(ys, tuple): ys = (ys,)  # if forward result is not a tuple
        outputs = [Variable(as_array(y)) for y in ys]  # prevented : numpy changes (0-dim ndarray) into (scalar) when it is computed
        
        if Config.enable_backprop:  # all for backpropagation
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]  # prevented : circular reference
        
        return outputs if len(outputs) > 1 else outputs[0]  # if result is (y,), just return y
    
    def forward(self, xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.inputs[0].data  # *inputs serves variable as tuple : (x,)
        return 2 * x * gy


class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.inputs[0].data
        return np.exp(x) * gy

class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1
    
    def backward(self, gy):
        return gy, gy

# rapper functions
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def add(x0, x1):
    return Add()(x0, x1)

def no_grad():
    return using_config('enable_backprop', False)

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


for i in range(10):
    x = Variable(np.random.randn(10000))
    y = square(square(square(x)))