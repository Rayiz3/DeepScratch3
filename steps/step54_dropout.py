if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # to find 'dezero' directory wherever execute a .py file
    
import numpy as np
import dezero.functions as F
from dezero import test_mode


x = np.ones(5)
print(x)

y = F.dropout(x)
print('train:', y)

with test_mode():
    y = F.dropout(x)
    print('test:', y)