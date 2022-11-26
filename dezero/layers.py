import os
import numpy as np
import dezero.functions as F
from dezero import cuda
from dezero.utils import pair
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
    
    def to_cpu(self):
        for param in self.params():
            param.to_cpu()
    
    def to_gpu(self):
        for param in self.params():
            param.to_gpu()
    
    def _flatten_params(self, params_dict, parent_key=''):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name
            
            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj
    
    def save_weight(self, path):
        self.to_cpu()
        
        params_dict={}
        self._flatten_params(params_dict)  # {key : Variable}
        array_dict = {key : param.data for key, param in params_dict.items() if param is not None}  # {key : ndarray}
        
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:  # exceptional flow for user interrupt (ctrl + c)
            if os.path.exists(path):
                os.remove(path)
            raise
    
    def load_weight(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]
    
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
    
class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1, pad=0, nobias=False, dtype=np.float32, in_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype
        
        # parameter initilize - W
        self.W = Parameter(None, name='W')
        if in_channels is not None:  # suspend to forward() if in_size is not determined
            self._init_W()
        # parameter initilize - b
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')
    
    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        self.W.data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
    
    def forward(self, x):
        if self.W.data is None:  # when it is suspended
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)
        
        return F.functions_conv.conv2d(x, self.W, self.b, self.stride, self.pad)

class RNN(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        self.x2h = Linear(hidden_size, in_size=in_size)  # use only one bias
        self.h2h = Linear(hidden_size, in_size=in_size, nobias=True)  # thus it does not necessary
        self.h = None
        
    def reset_state(self):
        self.h = None
    
    def forward(self, x):
        if self.h is None:
            h_new = F.tanh(self.x2h(x))
        else:
            h_new = F.tanh(self.h2h(self.h) + self.x2h(x))
        self.h = h_new
        return h_new

class LSTM(Layer):
    def __init__(self, hidden_size, in_size=None):
        super().__init__()
        
        H, I  = hidden_size, in_size
        self.x2f = Linear(H, in_size=1)
        self.h2f = Linear(H, in_size=H, nobias=True)
        self.x2i = Linear(H, in_size=1)
        self.h2i = Linear(H, in_size=H, nobias=True)
        self.x2o = Linear(H, in_size=1)
        self.h2o = Linear(H, in_size=H, nobias=True)
        self.x2u = Linear(H, in_size=1)
        self.h2u = Linear(H, in_size=H, nobias=True)
        
        self.reset_state()
    
    def reset_state(self):
        self.h = None
        self.c = None
    
    def forward(self, x):
        if self.h is None:
            f = F.sigmoid(self.x2f(x))
            i = F.sigmoid(self.x2i(x))
            o = F.sigmoid(self.x2o(x))
            u = F.tanh(self.x2u(x))
        else:
            f = F.sigmoid(self.x2f(x) + self.h2f(self.h))
            i = F.sigmoid(self.x2i(x) + self.h2i(self.h))
            o = F.sigmoid(self.x2o(x) + self.h2o(self.h))
            u = F.tanh(self.x2u(x) + self.h2u(self.h))
        
        if self.c is None:
            c_new = i * u
        else :
            c_new = f * self.c + i * u
        
        h_new = o * F.tanh(c_new)
        
        self.h, self.c = h_new, c_new
        return h_new