import numpy as np
import scipy
import BioKIT as bk

from . import Node
from . import FrameBuffer
from . import LambdaNode

class F0Calculator(Node.Node):
    """
    Takes a continuous stream of data as input and outputs a spectrogram.
    """
    def __init__(self, frame_length_ms, frame_shift_ms, sample_rate, channel = 0, f0_min = 95.0, f0_max = 300.0, harmo_thresh = 0.2, warm_start = True, name = "F0Calculator"):
        """Initializes all the buffers used to run-on calculate F0s."""        
        self.channel = channel
        self.sr = sample_rate
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.harmo_thresh = harmo_thresh
        
        # Set up subgraph
        self.channel_sel = LambdaNode.LambdaNode(lambda x, sel_channel = self.channel: x[:,[sel_channel]],  name = name + ".ChannelSel")
        self.audio_fb = FrameBuffer.FrameBuffer(frame_length_ms, frame_shift_ms, sample_rate, name = name + ".FrameBuffer", warm_start = warm_start)(self.channel_sel)
        self.f0_calc = LambdaNode.LambdaNode(self.frame_f0s,  name = name + ".F0Calc")(self.audio_fb)
        
        # Set up subgraph passthrough.
        self.set_passthrough(self.channel_sel, self.f0_calc)
        
    def frame_f0s(self, frame):
        """
        Extract frame pitch using the YIN algorithm. Based on an implementation by
        Patrice Guyot ( https://github.com/patriceguyot/Yin/blob/master/yin.py )
        
        Make sure that the passed number of samples is large enough for the desired
        f0_max and also a power of two.
        """
        frame = frame.flatten()
        w = frame.size
        
        tau_min = int(self.sr / self.f0_max)
        tau_max = int(self.sr / self.f0_min)
        
        # Diff. func
        x = np.array(frame, np.float64)
        x_cumsum = np.concatenate((np.array([0.]), (x * x).cumsum()))
        size_pad = w + w // 2
        fc = np.fft.rfft(x, size_pad)
        conv = np.fft.irfft(fc * fc.conjugate())[:tau_max]
        df = x_cumsum[w:w - tau_max:-1] + x_cumsum[w] - x_cumsum[:tau_max] - 2 * conv
        
        # Cumulative mean normalized difference
        cmndf = df[1:] * range(1, tau_max) / np.cumsum(df[1:]).astype(float)
        cmndf = np.insert(cmndf, 0, 1)
        
        # Pitch extraction
        tau = tau_min
        while tau < tau_max:
            if cmndf[tau] < self.harmo_thresh:
                while tau + 1 < tau_max and cmndf[tau + 1] < cmndf[tau]:
                    tau += 1
                return np.array([self.sr / tau])
            tau += 1
        
        # If we got here, no pitch was found
        return np.array([0.0])
