import numpy as np
import ctypes
import os
import copy

import time

from . import Node

def smooth_clamp(x, mi, mx): 
    return mi + (mx-mi)*(lambda t: np.where(t < 0 , 0, np.where( t <= 1 , 3*t**2-2*t**3, 1 ) ) )( (x-mi)/(mx-mi) )

# softclamp the 0th bark coefficient to make silence more silent and eliminate hard clipping
def b0_clamp(x):
    return smooth_clamp(x, -5.0, 15.0)

class LPCNetSynth(Node.Node):
    """
    Takes a continuous stream of LPCNet feature vectors as input, produces chunks of audio as output
    Only supports 10ms shift 16kHz audio
    """
    def __init__(self, name = "LPCNetSynth"):
        """Initializes the vocoder"""
        super(LPCNetSynth, self).__init__(name = name)
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.lpcnet = ctypes.cdll.LoadLibrary(os.path.join(base_path, "..", "LPCNet", "lpcnet.so"))
                
        self.c_feats = (ctypes.c_float * 55)()
        self.c_retval = (ctypes.c_short * 160)()
        
        self.feat_indices = list(range(0, 18)) + [36, 37]
        self.zero_indices = set(range(0, 55)) - set(self.feat_indices)
        
        # Try to make sure the encoder does not break
        #self.minv =[-4.2393956 , -7.2886868 , -5.092843  , -0.95322895, -4.368583  , -1.7447604 , -2.612796  , -1.3579264 , -2.179475  , -1.6134926 , -2.3843174 , -0.5269048 , -1.0468249 , -0.58243126, -0.8876593 , -0.42780092, -0.6185482 , -0.6301454]
        #self.maxv = [14.620428  ,  5.32089   ,  2.1931405 ,  5.3950467 ,  1.4269606 , 1.8303461 ,  0.26465794,  1.1120001 ,  0.2858516 ,  1.4064349 , -0.1431984 ,  1.2534863 ,  0.6931837 ,  0.7419731 ,  0.74293303, 0.6841681 ,  0.5422645 ,  0.45173505]
        
        self.reset()
        #if warm_start == True:
        #    self.warm_start = 30
        #else:
        #    self.warm_start = 0
        
    def reset(self):
        """Resets vocoder state"""
        self.lpcnet.moznet_reset_decoder()
        for idx in self.zero_indices:
            self.c_feats[idx] = 0.0
    
    def add_data(self, data_frame, data_id):
        #print("LPCNEt add", data_frame.shape)
        """Add frames (18 bark-scale cepstral coefficients + cont. f0 + aperiodicity) and perform vocoding."""
        if len(data_frame.shape) == 1:
            data_frame = [data_frame]
        
        #data_frame[:, 0] = b0_clamp(data_frame[:, 0]) * 0.95
        #print(data_frame[:, 0].shape)
        #data_frame[:, 0] = np.mean(data_frame[:, 1:18], axis = 1) * 3.0
        data_frame[:, 0] = (data_frame[:, 0] - 5.0) * 1.2 # small preemph
        data_frame = np.clip(data_frame, -20, 20) # Ensure encoder doesn't choke
        #data_frame[:, 0] = np.maximum(-5.0, np.minimum(data_frame[:, 0], 15.0))
        #if self.warm_start > 0:
            #self.warm_start -= 1
        #else:
        #data_frame[:, 0] = (data_frame[:, 0] / 20.0)
        #data_frame[:, 0] = np.mean(np.square(data_frame[:, 1:18]), axis=1) * 6.0 - 5.0

        for frame in data_frame:
            for (idx, feature) in zip(self.feat_indices, frame):
            #    if idx < 18:
            #        feature = smooth_clamp(feature, self.minv[idx], self.maxv[idx])
                self.c_feats[idx] = feature
            try:
                #print("Try synth")
                self.lpcnet.moznet_decode(ctypes.byref(self.c_feats), ctypes.byref(self.c_retval))
                return_data = copy.deepcopy(np.nan_to_num(np.array(self.c_retval))) * 0.95
                #print(np.max(return_data))
                self.output_data(np.int16(np.clip(return_data, -0.98 * (2**15 - 1), 0.98 * (2**15 - 1))))
            except Exception as e:
                print("Exception in LPCNet thread (can be further down computation graph!):", e)
                
