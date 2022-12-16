import numpy as np
from collections import deque
from itertools import islice
from bisect import bisect_left, insort

import time
from . import Node

class RunningNorm(Node.Node):
    """
    Takes a stream of frames and normalize them (per dimension) using a
    running percentile approach
    """
    def __init__(self, channels = None, history_length = 550, max_divisor = 0.1, upper_percentile = 0.99, lower_percentile = 0.9, clip = True, update_every = 1, name = "RunningNorm"):
        """Stores parameters for the normalizing."""
        super(RunningNorm, self).__init__(name = name)
        
        self.history_length = history_length
        self.max_divisor = max_divisor
        self.upper_idx = int((self.history_length - 1) * upper_percentile)
        self.lower_idx = int((self.history_length - 1) * lower_percentile)
        self.clip = clip

        self.return_sample = None
        self.norm_lists = None
        self.sample_buffers = None
        self.channels = channels
        self.update_every = update_every
        self.update_counter = self.update_every - 1
        
    def add_data(self, data, data_id):
        """
        Running normalization of n-dimensional samples
        
        expects 2D data. Does not accept multiple inputs, crashes if you try, probably.
        """
        if self.return_sample is None:
            self.return_sample = []
            self.norm_lists = []
            self.sample_buffers = []
            
            if self.channels is None:
                self.channels = list(range(data.shape[1]))
                
            for i in range(data.shape[1]):
                self.return_sample.append(1.0)
                self.norm_lists.append(list(data[0, i].repeat(self.history_length)))
                self.sample_buffers.append([])
                
        data_return = []
        for sample in data:
            self.update_counter += 1
            if self.update_counter == self.update_every:
                self.update_counter = 0
                for i in self.channels:
                    sample_abs = np.abs(sample[i])
                    self.sample_buffers[i].append(sample[i])
                    self.sample_buffers[i] = self.sample_buffers[i][-self.history_length:]
                    old_elem = self.sample_buffers[i][0]
                    del self.norm_lists[i][bisect_left(self.norm_lists[i], old_elem)]
                    insort(self.norm_lists[i], sample_abs)
                    self.return_sample[i] = max(self.max_divisor, self.norm_lists[i][self.upper_idx])
                    #if self.return_sample[i] < self.norm_lists[i][self.lower_idx]: TODO?
                    #    self.return_sample[i] = 1.0
            data_return.append(sample / self.return_sample)
            
        if self.clip == True:
            self.output_data(np.clip(np.array(data_return), -1.0, 1.0))
        else:
            self.output_data(np.array(data_return))
