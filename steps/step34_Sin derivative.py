if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # to find 'dezero' directory wherever execute a .py file

import numpy as np
from dezero import Variable
import dezero.functions as F
import matplotlib.pyplot as plt


x = Variable(np.linspace(-7, 7, 200))
y = F.sin(x)
y.backward(create_graph=True)

logs = [y.data]

for i in range(3):
    gx = x.grad
    logs.append(gx.data)
    
    x.cleargrad()
    gx.backward(create_graph=True)
    # print(i, '-th derivative:', x.grad)

# plotting
labels = ["y=sin(x)", "y(1)", "y(2)", "y(3)"]
for i, v in enumerate(logs):
    plt.plot(x.data, logs[i], label=labels[i])
plt.legend(loc="lower right")
plt.show()