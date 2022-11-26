import math
import random
import numpy as np
from dezero import cuda


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, gpu=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)
        self.gpu = gpu
        
        self.reset()
    
    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else :
            self.index = np.arange(len(self.dataset))
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        # take batch
        xp = cuda.cupy if self.gpu else np
        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size:(i + 1) * batch_size]  # 'batch_size' data
        batch = [self.dataset[i] for i in batch_index]
        batch_x = xp.array([element[0] for element in batch])
        batch_t = xp.array([element[1] for element in batch])
        
        # update
        self.iteration += 1
        
        return batch_x, batch_t
    
    def next(self):
        return self.__next__()

class SeqDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, gpu=False):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=False, gpu=gpu)  # serial data => do not shuffle
    
    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration
    
        jump = self.data_size // self.batch_size  # number of batch
        batch_index = [(i * jump + self.iteration) % self.data_size for i in range(self.batch_size)]  # [j n+j 2n+j ... (B-1)n+j]
        batch = [self.dataset[i] for i in batch_index]
        
        xp = cuda.array if self.gpu else np
        x = xp.array([data[0] for data in batch])
        t = xp.array([data[1] for data in batch])
        
        self.iteration += 1
        return x, t