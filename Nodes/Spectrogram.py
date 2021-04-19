import numpy as np
from scipy import hanning
from . import Node
from . import FrameBuffer
from . import LambdaNode
import logging
import samplerate
import local.MelFilterBank as mel


logger = logging.getLogger('Spectrogram.py')


class LogMelSpectrogramConverter(Node.Node):
    """
    Node for extracting logMel spectral coefficients from an incoming audio stream. The Node is
    intended to work with a sample frequency of 16kHz and therefore embeds of an additional
    resampler to downsample the incoming audio stream.
    """

    def __init__(self, window_size, window_shift, sfreq, nb_bins=40, has_inputs=True, name='LogMelSpecConverter'):
        """
        Extract the LogMel spectrogram from the incoming acoustic signal

        :param window_size: Length of one window in ms
        :param window_shift: Frame shift in ms
        :param sfreq: Sampling rate
        :param nb_bins: Number of Mel bins
        :param name: Node name
        """
        super(LogMelSpectrogramConverter, self).__init__(name=name, has_inputs=has_inputs)

        self.nb_bin = 40
        self.downsampling_factor = sfreq / 16000
        self.window_function = hanning(int(window_size / 1000 * sfreq / self.downsampling_factor))
        self.mfb = mel.MelFilterBank(int(len(self.window_function) // 2 + 1), nb_bins, sfreq / self.downsampling_factor)

        logger.info('Framelength in ms: {:d}'.format(window_size))
        logger.info('Frameshift in ms: {:d}'.format(window_shift))
        logger.info('Samplerate: {:d}'.format(sfreq))

        # Set up subgraph
        self.segmenter = FrameBuffer.FrameBuffer(window_size, window_shift, warm_start=True,
                                                 sample_rate=sfreq, name=name + '.FrameBuffer')

        self.downsampler = LambdaNode.LambdaNode(self.resample_func, name=name+'.Downsampling')(self.segmenter)
        self.log_mel_calculator = LambdaNode.LambdaNode(self.compute_log_mels, name=name + ".LogMels")(self.downsampler)

        # Set up subgraph passthrough.
        self.set_passthrough(self.segmenter, self.log_mel_calculator)

    def resample_func(self, x):
        return self.resample_segment(x)

    def resample_segment(self, segment):
        """
        Resample the incoming audio stream to match 16kHz.
        :param segment: segment of the acoustic signal
        :return: 1D numpy array of the resampled signal
        """
        segment = samplerate.resample(segment, 1 / self.downsampling_factor, 'sinc_fastest')

        if len(segment) < len(self.window_function):
            idx = len(self.window_function) - len(segment)
            tmp = np.zeros((len(self.window_function), 1))
            tmp[idx:] = segment
            segment = tmp

        return segment.flatten()

    def compute_log_mels(self, frame):
        """
        Compute the LogMel-spectrogram of a given single segment of audio data
        :param frame: segment of an acoustic signal
        :return: LogMel representation of the segment
        """
        spec = np.fft.rfft(self.window_function * frame)
        spec = self.mfb.toLogMels(np.abs(spec))

        return spec
