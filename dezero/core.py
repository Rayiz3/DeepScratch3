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
    __array_priority__ = 200  # prevented : for operation with (ndarray * Variable), ndarray.__mul__ is applied at first
    
    def __init__(self, data, name=None):
        # exception flow
        if (data is not None) and (not isinstance(data, np.ndarray)):
            raise TypeError("type {} is not supported.".format(type(data)))
        
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0
    
    def __len__(self):  # len()
        return len(self.data)
    
    def __repr__(self):  # print()
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('/n', '/n' + ' ' * 9)
        return 'variable(' + p + ')'
    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def backward(self, retain_grad=False, create_graph=False):  # create_graph=False : do backpropagation only 'once'
        if self.grad is None: # if terminal variable (loss fuction value)
            self.grad = Variable(np.ones_like(self.data)) # default grad = 1s (with same DT with self.data)
        
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
            
            with using_config('enable_backprop', create_graph):  # mode switching inside function
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
    
    @property  # can use fuction as variable : x.shape
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
            

class Function:
    def __call__(self, *inputs):  # __call__ : called when f(...) (f is an instance of the class)
        inputs = [as_variable(x) for x in inputs] # to make (ndarray * Variable) operation available
        
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # unpacking : f([x0, x1]) => f(x0, x1)
        if not isinstance(ys, tuple): ys = (ys,)  # if forward result is not a tuple
        outputs = [Variable(as_array(y)) for y in ys]  # prevented : numpy changes (0-dim ndarray) into (scalar) when it is computed
        
        if Config.enable_backprop:  # all for backpropagation
            self.generation = max([x.generation for x in inputs])
            
            # make relation
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]  # prevented : circular reference
        
        return outputs if len(outputs) > 1 else outputs[0]  # if result is (y,), just return y
    
    def forward(self, xs):
        raise NotImplementedError()
    
    def backward(self, gys):
        raise NotImplementedError()

class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy

class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1
    
    def backward(self, gy):
        return gy, gy

class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1
    
    def backward(self, gy):
        return gy, -gy
    
class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1
    
    def backward(self, gy):
        x0, x1 = self.inputs
        return gy * x1, gy * x0
    
class Div(Function):
    def forward(self, x0, x1):
        return x0 / x1
    
    def backward(self, gy):
        x0, x1 = self.inputs
        return gy / x1, gy * (-x0 / x1 ** 2)
    
class Pow(Function):
    def __init__(self, c):
        self.c = c
         
    def forward(self, x):
        return x ** self.c
    
    def backward(self, gy):
        x = self.inputs[0]
        c = self.c
        return c * x ** (c - 1) * gy


# rapper functions
def neg(x):
    return Neg()(x)

def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)

def pow(x, c):
    return Pow(c)(x)

# mode switching
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

# change (othertype) => (Variable)
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

def setup_variable():
    Variable.__neg__ = neg
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow