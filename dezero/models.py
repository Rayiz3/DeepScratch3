from dezero import Layer
from dezero import utils


class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = sefl.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)