if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # to find 'dezero' directory wherever execute a .py file

import numpy as np
from dezero import Variable
import dezero.functions as F


# toy dataset
np.random.seed(0)  # fix random seed
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# initialization
I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(I, H))  # I * H matrix
b1 = Variable(np.zeros(H))  # H * 1
W2 = Variable(0.01 * np.random.randn(H, O))  # H * O matrix
b2 = Variable(np.zeros(O))  # O * 1

def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y


lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y_pred, y)
    
    # backpropagation
    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()
    
    # gradient descent
    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    
    if i % 1000 == 0:
        print(loss)
    