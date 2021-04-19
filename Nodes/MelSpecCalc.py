import numpy as np 
from mne.filter import filter_data
import pyedflib
import scipy.io.wavfile as wav
import scipy
import scipy.signal
import MelFilterBank as mel
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.stats import mode

from . import Node
from . import FrameBuffer
from . import LambdaNode


class MelSpecCalc(Node.Node):
    """
    Takes a continuous stream of audio data as input and outputs mel-spectrum features.
    """
    def __init__(self, sample_rate, name = "MelSpecCalc"):
        """Initializes all the nodes used to calculate MelSpec features"""
        
        # Set variables up
        super().__init__(name = name)
        self.sample_rate = sample_rate
        
        # # Set up subgraph
        # self.channel_sel = LambdaNode.LambdaNode(lambda x, sel_channel = self.channel: x[:,[sel_channel]],  name = name + ".ChannelSel")
        # self.audio_fb = FrameBuffer.FrameBuffer(frame_len_ms, frame_shift_ms, sample_rate, name = name + ".FrameBuffer")(self.channel_sel)
        # self.wcep_calc = LambdaNode.LambdaNode(self.frame_wceps,  name = name + ".WCEPCalc")(self.audio_fb)
        
        # # Set up subgraph passthrough.
        # self.set_passthrough(self.channel_sel, self.wcep_calc)
        
        
    def extract_mel_specs(self,frame,resample_rate=16000):
        #Process audio
        audio = scipy.signal.decimate(frame,int(self.sample_rate/resample_rate))
        scaled = np.int16(frame/np.max(np.abs(frame)) * 32767) #32767?
        
        #calc LogMelSpec
        win = scipy.hanning(np.floor(window_length*sr + 1))[:-1]
        spectrogram = np.zeros((num_windows, int(np.floor(window_length*sr / 2 + 1))),dtype='complex')
        
        spectrogram = np.fft.rfft(win*frame)
        
        mfb = mel.MelFilterBank(spectrogram.shape[1], 23, self.sample_rate)
        spectrogram = np.abs(spectrogram)
        spectrogram = (mfb.toLogMels(spectrogram)).astype('float')
        return spectrogram
        
   
