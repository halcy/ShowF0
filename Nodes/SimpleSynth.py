import numpy as np
import scipy
import scipy.signal as signal
import random
import math
import ctypes
import os
import copy

import time

from . import Node
from . import FrameJoinStacker
from . import LambdaNode

class SimpleSynth(Node.Node):
    """
    Takes a continuous volume stream (0 -> 1) and generates buzzing at the specified volume and given frequency, optionally mixed with noise.
    """
    def __init__(self, frame_shift_ms, sample_rate, buzz_freq = 150, noise_mix = 0.1, name = "SimpleSynth", quantized=True):
        """Initializes the synth."""
        super(SimpleSynth, self).__init__(name = name)
        
        # Make sure no integer accidents happen
        frame_shift_ms = float(frame_shift_ms)
        sample_rate = float(sample_rate)
        
        # Set variables up
        self.frame_shift_ms = frame_shift_ms        
        self.sample_rate = sample_rate
        self.sample_count = 0
        self.buzz_freq = buzz_freq
        self.noise_mix = noise_mix
        self.quantized = quantized
        
    def add_data(self, data_frame, data_id):
        for data in data_frame:
            out_len = int((self.sample_rate / 1000.0) * self.frame_shift_ms)
            
            if self.quantized == True:
                # linear one hot level (0, 1, 2) dequant
                data = np.argmax(data) * 0.5
            else:
                data = np.clip(data, 0.0, 1.0)
            #print(data.shape, data)
            
            sin_base = np.sin(np.array(list(range(self.sample_count, self.sample_count + out_len))) / self.sample_rate * math.pi * 2.0 * self.buzz_freq)
            rect_base = (sin_base > 0).astype(float)
            harm_base = (sin_base + rect_base) / 2.0
            out_frame = (harm_base * (1.0 - self.noise_mix) + np.random.random(size=(out_len)) * self.noise_mix) * data * 0.99 * (2**15)
            
            self.sample_count += out_len
            self.output_data(np.int16(np.clip(out_frame, -0.99 * (2**15 - 1), 0.99 * (2**15 - 1))))
            
