import numpy as np
import scipy
import BioKIT as bk

from . import Node
from . import FrameBuffer
from . import LambdaNode

class WCEPCalculator(Node.Node):
    """
    Takes a continuous stream of data as input and outputs a spectrogram.
    """
    def __init__(self, frame_len_ms, frame_shift_ms, sample_rate, num_wceps = 25, channel = 0, warm_start = True, name = "WCEPCalculator"):
        """Initializes all the nodes used to run-on calculate WCEP features."""
        super(WCEPCalculator, self).__init__(name = name)
        
        self.num_wceps = num_wceps
        self.warper = bk.CepstralWarper()
        self.channel = channel
        self.window = None
        
        # Set up subgraph
        self.channel_sel = LambdaNode.LambdaNode(lambda x, sel_channel = self.channel: x[:,[sel_channel]],  name = name + ".ChannelSel")
        self.audio_fb = FrameBuffer.FrameBuffer(frame_len_ms, frame_shift_ms, sample_rate, name = name + ".FrameBuffer", warm_start = warm_start)(self.channel_sel)
        self.wcep_calc = LambdaNode.LambdaNode(self.frame_wceps,  name = name + ".WCEPCalc")(self.audio_fb)
        
        # Set up subgraph passthrough.
        self.set_passthrough(self.channel_sel, self.wcep_calc)
        
    def frame_wceps(self, frame_in):
        """Returns warped MFCCs."""
        
        # Calculate blackman-windowed cepstrum
        if self.window is None:
            self.window = scipy.blackman(frame_in.shape[0]+1)[:-1]
            self.window /= np.sum(self.window)
        
        windowed_frame = self.window * frame_in.flatten()
        windowed_frame -= np.mean(windowed_frame)
        
        power_spec = np.fft.fft(windowed_frame)
        power_spec = np.real(power_spec) * np.real(power_spec) + np.imag(power_spec) * np.imag(power_spec)
        cepstrum = np.fft.ifft(np.log(power_spec + 0.00001))
        
        # Scale the first and the middle coefficients by 0.5
        cepstrum = np.real(cepstrum.reshape(1, cepstrum.shape[0]))
        cepstrum[:, 0] = cepstrum[:, 0] * 0.5
        cepstrum[:, len(self.window) // 2] = cepstrum[:, len(self.window) // 2] * 0.5
        
        warped_cepstrum_fs = self.warper.warpSequence([bk.FeatureSequence(cepstrum, True, True)], int(len(self.window) / 2), self.num_wceps, 0.42)[0] # TODO ???
        
        power_spec = power_spec.reshape(1, power_spec.shape[0])
        try:
            warped_cepstrum_fsOpt = self.warper.minimizeEstimationBias(warped_cepstrum_fs, bk.FeatureSequence(power_spec, True, True), 5, 0.42)
        except ValueError:
            warped_cepstrum_fsOpt = warped_cepstrum_fs
            
        return(warped_cepstrum_fs.getMatrix().flatten())
    
