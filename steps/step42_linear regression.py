if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # to find 'dezero' directory wherever execute a .py file

import numpy as np
from dezero import Variable
from dezero import functions as F

# toy dataset
np.random.seed(0)  # fix random seed
x = np.random.rand(100, 1)
y = 5 + 2*x + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

W = Variable(np.zeros((1,1)))
b = Variable(np.zeros(1))

def predict(x):
    return F.matmul(x, W) + b

def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)


lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y_pred, y)
    
    #initialize and backprogagate
    W.cleargrad()
    x.cleargrad()
    loss.backward()
    
    # gradient descent
    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    
    print(W, b , loss)