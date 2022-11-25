if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # to find 'dezero' directory wherever execute a .py file

import numpy as np
import dezero
import dezero.functions as F
import dezero.layers as L
import matplotlib.pyplot as plt
from dezero import Model


class SimpleRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = L.RNN(hidden_size)
        self.fc = L.Linear(out_size)
    
    def reset_state(self):
        self.rnn.reset_state()
    
    def forward(self, x):
        h = self.rnn(x)
        return self.fc(h)


# dummy time series data
seq_data = [np.random.randn(1, 1) for _ in range(1000)]
xs = seq_data[0:-1]
ts = seq_data[1:]  # let's predict the next data of x

# RNN for 2 times
model = SimpleRNN(10, 1)

loss, cnt = 0, 0

for x, t in zip(xs, ts):
    y = model(x)
    loss += F.mean_squared_error(y, t)
    
    cnt += 1
    if cnt == 2:
        model.cleargrads()
        loss.backward()
        break

# sin curve prediction (train)
train_set = dezero.datasets.SinCurve(train=True)
print(len(train_set))  # 999
print(train_set[0])  # (input data, label), xs[n] = tx[n-1]

# hyperparameter
max_epoch = 100
hidden_size = 100
bptt_length = 30
seqlen = len(train_set)

model = SimpleRNN(hidden_size, 1)
optimizer = dezero.optimizers.Adam().setup(model)
for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0
    
    for x, t in train_set:
        x = x.reshape(1, 1)
        
        # forward
        y = model(x)
        loss += F.mean_squared_error(y, t)
        count += 1
        
        # truncate - backward
        if count % bptt_length == 0 or count == seqlen:
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            
            # updated
            optimizer.update()
        
    avg_loss = float(loss.data) / count
    print('| epoch %d | loss %f' % (epoch + 1, avg_loss))

# cos curve predicton (test)
xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
model.reset_state()
pred_list = []

with dezero.no_grad():
    for x in xs:
        x = np.array(x).reshape(1, 1)
        y = model(x)
        pred_list.append(float(y.data))

plt.plot(np.arange(len(xs)), xs, label='y=cos(x)')
plt.plot(np.arange(len(xs)), pred_list, label='predict')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()