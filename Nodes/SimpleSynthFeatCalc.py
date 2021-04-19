import numpy as np
import scipy
import BioKIT as bk

from . import Node
from . import FrameBuffer
from . import LambdaNode

class SimpleSynthFeatCalc(Node.Node):
    """
    Takes a continuous stream of data as input and outputs signal power
    """
    def __init__(self, frame_len_ms, frame_shift_ms, sample_rate, channel = 0, warm_start = True, name = "SimpleSynthFeatCalc"):
        """Initializes all the nodes used to run-on calculate WCEP features."""
        super(SimpleSynthFeatCalc, self).__init__(name = name)
        
        self.channel = channel
        self.window = None
        
        # Set up subgraph
        self.channel_sel = LambdaNode.LambdaNode(lambda x, sel_channel = self.channel: x[:,[sel_channel]],  name = name + ".ChannelSel")
        self.audio_fb = FrameBuffer.FrameBuffer(frame_len_ms, frame_shift_ms, sample_rate, name = name + ".FrameBuffer", warm_start = warm_start)(self.channel_sel)
        self.wcep_calc = LambdaNode.LambdaNode(self.extract_feats,  name = name + ".WCEPCalc")(self.audio_fb)
        
        # Set up subgraph passthrough.
        self.set_passthrough(self.channel_sel, self.wcep_calc)
    
    def post_quant(self, data, level1 = 0.45, level2 = 0.8):
        """Linear quantization into three levels by quantiles"""
        labels = (data > np.quantile(data, level1)).astype(float) + (data > np.quantile(data, level2)).astype(float)
        labels = np.eye(3)[labels.flatten().astype(int)]
        return labels
    
    def extract_feats(self, frame_in):
        """Returns warped MFCCs."""
        
        # Calculate blackman-windowed cepstrum
        if self.window is None:
            self.window = scipy.blackman(frame_in.shape[0]+1)[:-1]
            self.window /= np.sum(self.window)
        
        windowed_frame = self.window * frame_in.flatten()
        windowed_frame -= np.mean(windowed_frame)
        
        result_frames = np.array([np.sum([x ** 2 for x in windowed_frame], axis = 0) / float(windowed_frame.shape[0])])
        return result_frames.reshape(-1, 1)
