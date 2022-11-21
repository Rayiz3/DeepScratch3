if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # to find 'dezero' directory wherever execute a .py file

import os
import dezero
import dezero.functions as F
from dezero import optimizers, DataLoader
from dezero.models import MLP


max_epoch = 3
batch_size = 100

train_set = dezero.datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)
model = MLP((1000, 10))
optimizer = optimizers.SGD().setup(model)

# load parameter
if os.path.exists('my_mlp.npz'):
    model.load_weight('my_mlp.npz')

for epoch in range(max_epoch):
    sum_loss = 0

    # dataset iteration(300)
    for x, t in train_loader:  # x : (1, 28, 28), t : 0 ~ 9
        # forward
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        
        # backward
        model.cleargrads()
        loss.backward()
        
        # updated
        optimizer.update()
        sum_loss += float(loss.data) * len(t)

    print('epoch: {}, loss: {:.4f}'.format(
        epoch + 1, sum_loss / len(train_set)))

# save parameter
model.save_weight('my_mlp.npz')