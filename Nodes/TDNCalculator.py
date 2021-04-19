import numpy as np
import math

from . import Node
from . import FrameBuffer
from . import LambdaNode
from . import FrameJoinStacker

import scipy.signal

class TDNCalculator(Node.Node):
    """
    Takes a continuous stream of data as input and framed TDN data
    """
    def __init__(self, frame_length_ms, frame_shift_ms, sample_rate, stacking_height, channels = None, warm_start = True, name = "TDNCalculator"):
        """Initializes all the buffers used to run-on calculate TDN features."""
        super(TDNCalculator, self).__init__(name = name)
        
        # Channel selector
        self.channels = channels
        if not self.channels is None:
            self.channel_sel = LambdaNode.LambdaNode(lambda x, sel_channels = self.channels: x[:,sel_channels], name = name + ".ChannelSel")
        else:
            self.channel_sel = LambdaNode.LambdaNode(lambda x: x, name = name + ".ChannelSel")
            
        # Set up filters
        #split_cutoff = 73.0 # The point at which a non-normalized 9-point double averaging reaches -3db
        split_cutoff = 134.0 # The point at which a non-normalized 9-point double averaging reaches -3db
        #split_cutoff = 175.0 # The point at which a non-normalized 9-point double averaging reaches -3db <- or was the good run this?
        line_frequency = 50.0 # Correct for EU
        filter_order = 3 # Lower for smaller group delay, this should result in 12ish samples max
        
        sos_notch_b, sos_notch_a = scipy.signal.iirnotch(w0 = line_frequency / (sample_rate / 2.0), Q=60.0)
        sos_notch = scipy.signal.tf2sos(sos_notch_b, sos_notch_a)        
        sos_lowpass = scipy.signal.iirfilter(ftype='butter', btype='lowpass', N=filter_order, Wn=split_cutoff / (sample_rate / 2.0), output="sos")
        sos_highpass = scipy.signal.iirfilter(ftype='butter', btype='highpass', N=filter_order, Wn=split_cutoff / (sample_rate / 2.0), output="sos")
        
        one_frame = (1.0 / sample_rate) * 1000.0
        self.notch_filter = FrameBuffer.FrameBuffer(one_frame, one_frame, sample_rate, sos_notch, name = name + ".LineNoiseRemoval")(self.channel_sel)
        #self.notch_filter = FrameBuffer.FrameBuffer(one_frame, one_frame, sample_rate, name = name + ".LineNoiseRemoval")(self.channel_sel)
        self.lowpass_fb = FrameBuffer.FrameBuffer(frame_length_ms, frame_shift_ms, sample_rate, sos_lowpass, name = name + ".LowpassFrameBuffer", warm_start = warm_start)(self.notch_filter)
        self.highpass_fb = FrameBuffer.FrameBuffer(frame_length_ms, frame_shift_ms, sample_rate, sos_highpass, name = name + ".HighpassFrameBuffer", warm_start = warm_start)(self.notch_filter)
    
        # 1st feature, power of low frequencies:        
        self.lowfreq_power = LambdaNode.LambdaNode(self.frame_power, name = name + ".LowFreqPower")(self.lowpass_fb)

        # 2nd feature, mean of low frequencies:        
        self.lowfreq_mean = LambdaNode.LambdaNode(self.frame_mean, name = name + ".LowFreqMean")(self.lowpass_fb)
        
        # 3rd feature, power of high frequencies:        
        self.highfreq_power = LambdaNode.LambdaNode(self.frame_power, name = name + ".HighFreqPower")(self.highpass_fb)

        # 4th feature, Zero crossing rate of high frequency band:        
        self.highfreq_zcr = LambdaNode.LambdaNode(self.frame_zero_crossings, name = name + ".HighFreqZCR")(self.highpass_fb)

        # 5th feature, mean of absolute values of high frequencies:        
        self.highfreq_abs_mean = LambdaNode.LambdaNode(self.abs_frame_mean, name = name + ".HighFreqAbsMean")(self.highpass_fb)

        # Create feature joiner / stacker
        self.join_stacker = FrameJoinStacker.FrameJoinStacker(stacking_height = stacking_height, name = name + ".JoinStacker", warm_start = warm_start)([
            self.lowfreq_power,
            self.lowfreq_mean,
            self.highfreq_power,
            self.highfreq_zcr,
            self.highfreq_abs_mean
        ])
        
        # Set passthrough
        self.set_passthrough(self.channel_sel, self.join_stacker)
            
    def frame_power(self, frame):
        """Takes frames from a buffer and returns their power."""
        return np.array([np.sum([x ** 2 for x in frame], axis = 0) / float(frame.shape[0])])

    def frame_mean(self, frame):
        """Takes frames from a buffer and returns their mean."""
        return np.array([np.mean(frame, axis = 0)])

    def abs_frame_mean(self, frame):
        """Takes frames from a buffer and returns the mean of their absolute values."""
        return np.array([np.mean(np.abs(frame), axis = 0)])

    def frame_zero_crossings(self, frame):
        """Calculates the zero crossing rate of a frame (0 counts as negative)."""
        return np.array([(np.diff(np.floor((np.sign(frame) + 2) / 2) - 1, axis = 0) != 0).sum(0) / float(frame.shape[0])])

