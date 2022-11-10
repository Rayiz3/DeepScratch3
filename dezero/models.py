from dezero import Layer
from dezero import utils

import dezero.functions as F
import dezero.layers as L


class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = sefl.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)
    
class MLP(Model):  # for n - layer affine model
    def __init__(self, fc_output_sizes, activation=F.sigmoid):  # fc : full connect
        super().__init__()
        self.activation = activation
        self.layers = []
        
        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l' + str(i), layer)  # self.ln = layer
            self.layers.append(layer)