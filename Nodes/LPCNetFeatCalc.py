import numpy as np
import ctypes
import os
import copy

import time

from . import Node
from . import FrameBuffer
from . import LambdaNode
    
class LPCNetFeatCalc(Node.Node):
    """
    Takes a continuous stream of audio samples as input and outputs LPCNet features.
    """
    def __init__(self, name = "LPCNetFeatCalc", warm_start = True):
        """Initializes the vocoder"""
        super(LPCNetFeatCalc, self).__init__(name = name)
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.lpcnet = ctypes.cdll.LoadLibrary(os.path.join(base_path, "..", "LPCNet", "lpcnet.so"))
                
        self.c_data = (ctypes.c_short * (160 * 4))()
        self.c_retval = (ctypes.c_float * (55 * 4))()
        
        self.feat_indices = list(range(0, 18)) + [36, 37]
        
        self.reset()
        
        self.audio_fb = FrameBuffer.FrameBuffer(40, 40, 16000, name = name + ".FrameBuffer")
        self.feat_calc = LambdaNode.LambdaNode(self.extract_feats,  name = name + ".FeatCalc")(self.audio_fb)
        
        # warm start manual (frame shift is _actually_ 10, lpcnet extractor just operates on 4 frames at a time)
        #self.warm_start = warm_start
        # This is nonsense, makes no sense to do that here, won't reduce lag
        #if warm_start == True:
        #    self.audio_fb.add_data(np.zeros((640 - 160, warm_start_dims)))
            
        self.warm_start = warm_start
        
        # Set up subgraph passthrough.
        self.set_passthrough(self.audio_fb, self.feat_calc)
        
    def reset(self):
        """Resets feature extractor state"""
        self.lpcnet.moznet_reset_encoder()
        for idx in range(55 * 4):
            self.c_retval[idx] = 0.0
    
    def extract_feats(self, audio_frame):
        """Add some audio data, extract and return features if sufficient data is present
        
           Integers -2**15 -> 2**15 please
        """
        for (idx, sample) in enumerate(np.int16(audio_frame[:, 0])):
            self.c_data[idx] = sample
        
        self.lpcnet.moznet_encode(ctypes.byref(self.c_data), ctypes.byref(self.c_retval))
        if self.warm_start == True:
            # Drop first frames to fix alignment -> costs us some data, but makes the algorithmic lag -5ms (!) in practice, and should be okay because of EMD
            self.warm_start = False
            #return np.array([]).reshape(0, len(self.feat_indices)) #copy.deepcopy(np.array(self.c_retval).reshape(4, 55)[3:, self.feat_indices])
            return copy.deepcopy(np.array(self.c_retval).reshape(4, 55)[3:, self.feat_indices])
        else:
            return copy.deepcopy(np.array(self.c_retval).reshape(4, 55)[:, self.feat_indices])
            
