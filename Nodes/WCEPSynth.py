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

class WCEPSynth(Node.Node):
    """
    Takes a continuous stream of MFCCs and potentially F0s as input, produces chunks of audio as output
    """
    def __init__(self, frame_shift_ms, sample_rate, mel_coeff_count, norm_factor = 1.0, feat_min = -1.0, feat_max = 1.0, use_f0s = True, name = "WCEPSynth"):
        """Initializes the synth."""
        super(WCEPSynth, self).__init__(name = name)
        
        # Make sure no integer accidents happen
        frame_shift_ms = float(frame_shift_ms)
        sample_rate = float(sample_rate)
        
        # Set variables up
        self.frame_shift_ms = frame_shift_ms        
        self.sample_rate = sample_rate
        self.norm_factor = norm_factor
        
        self.feat_min = feat_min
        self.feat_max = feat_max
        if not isinstance(self.feat_min, list):
            self.feat_min = [self.feat_min] * mel_coeff_count
        if not isinstance(self.feat_max, list):
            self.feat_max = [self.feat_max] * mel_coeff_count
            
        self.use_f0s = use_f0s
        self.mel_coeff_count = mel_coeff_count
        
        # Make c buffers, start up vocoder
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.synthesis = ctypes.cdll.LoadLibrary(os.path.join(base_path, "..", "Synthesis", "synthesis", "synthesis_runon.so"))
        self.c_sample_rate = ctypes.c_double(self.sample_rate)
        self.c_frame_ms = ctypes.c_double(self.frame_shift_ms)
        self.c_mfcc_count = ctypes.c_int(self.mel_coeff_count - 1)
        self.synthesis.reset_vocoder(self.c_sample_rate, self.c_frame_ms, self.c_mfcc_count)
        
        self.c_f0 = ctypes.c_double(0.0)
        self.c_wceps = (ctypes.c_double * self.mel_coeff_count)()
        self.c_retval = (ctypes.c_double * 320)()
        self.c_retsize = ctypes.c_long(0)
        self.synthesis.mfcc2wav(ctypes.byref(self.c_wceps), self.c_f0, ctypes.byref(self.c_retval), ctypes.byref(self.c_retsize))
        
        # Set up subgraph
        self.joiner = FrameJoinStacker.FrameJoinStacker(name = name + ".Joiner", warm_start = False)
        self.synth = LambdaNode.LambdaNode(self.call_synth, name = name + ".Synth")(self.joiner)
        self.set_passthrough(self.joiner, self.synth)
        
    def call_synth(self, data_frame):
        """Add a single frame of data, process it and call callbacks."""
        # TODO this should also work with 2D data instead of single frames
        if self.use_f0s:
            f0val = max(0.0, data_frame[1][0])
            if f0val < 70.0:
                f0val = 0.0
            if f0val > 350.0:
                f0val = 350.0
                
        data_frame = data_frame[0][0].flatten()
        
        for (idx, wcep) in enumerate(data_frame):
            #print("Index:",idx)
            #print("vorher self.c_wceps[idx]:",self.c_wceps[idx])

            self.c_wceps[idx] = float(min(max(wcep, self.feat_min[idx]), self.feat_max[idx]))
            #print("nachher self.c_wceps[idx]:",self.c_wceps[idx])
            
        if self.use_f0s:
            self.c_f0.value = f0val
        else:
            self.c_f0.value = 0.0
        
        self.synthesis.mfcc2wav(ctypes.byref(self.c_wceps), self.c_f0, ctypes.byref(self.c_retval), ctypes.byref(self.c_retsize))

        return_buffer = [] # TODO static alloc
        
        if self.c_retsize.value > 0:
            for idx in range(0, self.c_retsize.value):
                return_buffer.append(self.c_retval[idx])
            
            return_data = np.array(return_buffer)
            if not np.all(np.isfinite(return_data)):
                #print("Detected NaN, resetting vocoder.")
                return_data = copy.deepcopy(np.nan_to_num(return_data, copy = True))
                self.synthesis.reset_vocoder(self.c_sample_rate, self.c_frame_ms, self.c_mfcc_count)
                
                self.c_f0 = ctypes.c_double(0.0)
                self.c_wceps = (ctypes.c_double * self.mel_coeff_count)()
                self.c_retval = (ctypes.c_double * 320)()
                self.c_retsize = ctypes.c_long(0)
                self.synthesis.mfcc2wav(ctypes.byref(self.c_wceps), self.c_f0, ctypes.byref(self.c_retval), ctypes.byref(self.c_retsize))
                
            # return_data /= (2**15) / 10.0 # why
            # print(np.min(return_data), np.max(return_data))
            return_data = np.int16(np.clip(return_data * (2**15 - 1) / (self.norm_factor * 1.01), -0.99 * (2**15 - 1), 0.99 * (2**15 - 1)))
            
            return return_data
            
