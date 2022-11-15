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
from dezero.datasets import Spiral
from dezero.models import MLP

# normalize
n = transforms.Normalize(mean=0.0, std=2.0)
train_set = dezero.datasets.Spiral(train=True, transform=n)
print(train_set[0], len(train_set))

# take mini batch
batch_index = [0, 1, 2]
batch = [train_set[i] for i in batch_index]

x = np.array([element[0] for element in batch])
t = np.array([element[1] for element in batch])

# hyperparameter
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

# initilization
train_set = Spiral(train=True)  # (300, 2) / (300,)
test_set = Spiral(train=False)  # (300, 2) / (300,)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)
data_size = len(x)  # 300
max_iter = math.ceil(data_size / batch_size) # 300 / 30 = 10

for epoch in range(max_epoch):
    print('epoch: {}'.format(epoch+1))
    
    ## train section ##
    index = np.random.permutation(data_size)  # shuffle
    sum_loss, sum_acc = 0, 0
    
    # dataset iteration(300)
    for x, t in train_loader:
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
        print('train loss: {:.4f}, accuracy: {:.4f}'.format(avg_loss, avg_acc))