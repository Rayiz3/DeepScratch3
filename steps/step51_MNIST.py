if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # to find 'dezero' directory wherever execute a .py file

import math
import numpy as np
import dezero
import dezero.functions as F
from dezero import optimizers
from dezero import transforms
from dezero import DataLoader
from dezero.datasets import MNIST
from dezero.models import MLP

# hyperparameter
max_epoch = 5
batch_size = 100
hidden_size = 1000

# initilization
train_set = MNIST(train=True) # 60000
test_set = MNIST(train=False) # 10000
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
optimizer = optimizers.SGD().setup(model)

for epoch in range(max_epoch):
    print('epoch: {}'.format(epoch+1))
    
    ## train section ##
    sum_loss, sum_acc = 0, 0
    
    # dataset iteration(300)
    for x, t in train_loader:  # x : (1, 28, 28), t : 0 ~ 9
        # forward
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        
        # backward
        model.cleargrads()
        loss.backward()
        
        # updated
        optimizer.update()
        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)
    
    avg_loss = sum_loss / len(train_set)
    avg_acc = sum_acc / len(train_set)
    print('train loss: {:.4f}, accuracy: {:.4f}'.format(avg_loss, avg_acc))
    
    ## test section ##
    sum_loss, sum_acc = 0, 0
    
    with dezero.no_grad():
        for x, t in test_loader:
            # forward
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            
            # update
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
        
        avg_loss = sum_loss / len(test_set)
        avg_acc = sum_acc / len(test_set)
        print('test loss: {:.4f}, accuracy: {:.4f}'.format(avg_loss, avg_acc))