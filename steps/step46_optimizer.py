if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # to find 'dezero' directory wherever execute a .py file

import numpy as np
from dezero import Variable, Model
import dezero.functions as F
import dezero.layers as L


# toy dataset
np.random.seed(0)  # fix random seed
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# hyperparameter
lr = 0.2
iters = 10000
hidden_size = 10

# initialization
class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)
        
    def forward(self, x):
        return self.l2(F.sigmoid(self.l1(x)))


model = TwoLayerNet(hidden_size, 1)

for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y_pred, y)
    
    # backpropagation
    model.cleargrads()
    loss.backward()
    
    # gradient descent
    for p in model.params():
        p.data -= lr * p.grad.data
    
    if i % 1000 == 0:
        print(loss)
    