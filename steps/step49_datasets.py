if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # to find 'dezero' directory wherever execute a .py file

import numpy
import dezero
import dezero.functions as F
from dezero import optimizers
from dezero.models import MLP


train_set = dezero.datasets.Spiral(train=True)
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
x, t = dezero.datasets.get_spiral(train=True)  # (300, 2) / (300,)
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)
data_size = len(x)  # 300
max_iter = math.ceil(data_size / batch_size) # 300 / 30 = 10

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)  # shuffle
    sum_loss = 0
    
    # dataset iteration(300)
    for i in range(max_iter):
        batch_index = index[i * batch_size:(i + 1) * batch_size]  # 'batch_size' data
        batch_x = x[batch_index]  # ndarray
        batch_t = t[batch_index]  # ndarray
    
        # forward
        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        
        # backward
        model.cleargrads()
        loss.backward()
        
        # updated
        optimizer.update()
        sum_loss += float(loss.data) * len(batch_t)
    
    avg_loss = sum_loss / data_size
    print('epoch %d, loss %.2f' % (epoch+1, avg_loss))