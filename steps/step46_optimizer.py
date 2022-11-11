if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # to find 'dezero' directory wherever execute a .py file

import numpy as np
from dezero import Variable, optimizers
import dezero.functions as F
from dezero.models import MLP


# toy dataset
np.random.seed(0)  # fix random seed
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# hyperparameter
lr = 0.2
iters = 10000
hidden_size = 10

# initialization
model = MLP((hidden_size, 1))
optimizer = optimizers.SGD(lr).setup(model)

for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y_pred, y)
    
    # backpropagation
    model.cleargrads()
    loss.backward()
    
    # gradient descent
    optimizer.update()
    
    if i % 1000 == 0:
        print(loss)
    