import numpy as np
import dezero.functions as F
from dezero.core import Parameter
import weakref


class Layer:
    def __init__(self):
        self._params = set()
    
    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)
        
    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):  # if forward result is not a tuple
            outputs = (outputs,)
        
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]  # if result is (y,), just return y
    
    def forward(self, inputs):
        raise NotImplementedError()
    
    def params(self):
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):  # call parameters in the layer
                yield from obj.params()  # yield from : when yield a value using another yield-function(=generator)
            else:
                yield obj
    
    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()  # call parent class' __init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype
        
        # parameters(W, b)
        self.W = Parameter(None, name='W')
        if self.in_size is not None:  # suspend to forward() if in_size is not determined
            self._init_W() # make W
        
        self.b = None if nobias else Parameter(np.zeros(out_size, dtype=dtype), name='b')
        
    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1/I)
        self.W.data = W_data
    
    def forward(self, x):
        if self.W.data is None:  # when it is suspended
            self.in_size = x.shape[1]
            self._init_W()  # make W
        
        return F.linear(x, self.W, self.b)