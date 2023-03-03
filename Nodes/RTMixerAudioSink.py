import rtmixer
import sounddevice as sd
import multiprocessing
import numpy as np
import samplerate as sr
import math
import time

"""
RTMixer based low latency audio source for MS Windows
"""
from . import Node

class RTMixerAudioSink(Node.Node):
    def __init__(self, device_name_filter = None, sample_rate = 16000, exclusive_mode = False, name = "RTMixerAudioSink"):
        super(RTMixerAudioSink, self).__init__(has_inputs = True, name = name)

        # Parameters
        self.frame_pipe_out, self.frame_pipe_in = multiprocessing.Pipe(False)
        self.device_name_filter = device_name_filter
        self.sample_rate = sample_rate

        # Exclusive mode setting
        self.exclusive_mode = None
        if exclusive_mode:
            self.exclusive_mode = sd.WasapiSettings(exclusive=True)

        # Wind-down signaling
        self.stop_process = multiprocessing.Value('b', True)
  
    def find_device_and_rate(self):
        # Figure out WASAPI id
        wasapi_id = None
        wasapi_default_device = None
        for id, api in enumerate(sd.query_hostapis()):
            if "wasapi" in api["name"].lower():
                wasapi_id = id
                wasapi_default_device = api["default_output_device"]
                break

        # If device name filter is set, find device. Otherwise, use default.
        device_id = None
        if self.device_name_filter is None:
            device_id = wasapi_default_device
        else:
            for id, device in enumerate(sd.query_devices()):
                if not device["hostapi"] == wasapi_id or device["max_output_channels"] <= 0:
                    continue
                if self.device_name_filter.lower() in device["name"].lower():
                    device_id = id
                    break
        
        # See if we can get the desired sample rate directly and without resampling
        device_sample_rate = self.sample_rate
        try:
            sd.check_output_settings(
                device = device_id, 
                channels = 1, 
                dtype = 'float32', 
                extra_settings = self.exclusive_mode, 
                samplerate = device_sample_rate
            )
        except:
            # Doesn't work, have to resample
            device_sample_rate = sd.query_devices(device_id)["default_samplerate"]
        return device_id, device_sample_rate
    
    def add_data(self, data_frame, data_id=0):
        """Just send data to player process"""
        self.frame_pipe_in.send(data_frame)

    def audio_runner(self):
        """Process for shoving data at your ears"""
        # Get device and rate
        device, device_sample_rate = self.find_device_and_rate()
        sample_multiplier = device_sample_rate / self.sample_rate
        print("[Sink] Resampling with factor", sample_multiplier)

        # Open stream
        print("[Sink] Opening device: ", device, "@", device_sample_rate)
        stream = rtmixer.Mixer(
            device = device,
            channels = 1, 
            blocksize = 0, 
            samplerate = device_sample_rate,
            latency = 0,
            extra_settings = self.exclusive_mode
        )
        assert stream.dtype == 'float32'
        assert stream.samplesize == 4

        # Begin playback
        buffer_out = rtmixer.RingBuffer(4, 2 ** math.ceil(math.log2(device_sample_rate)))
        buffer_out.write(np.zeros((1024, 1), dtype='float32'))
        play_action = stream.play_ringbuffer(buffer_out, allow_belated = True)
        resampler = sr.Resampler('sinc_fastest', channels = 1)
        with stream:
            while self.stop_process.value == False:
                # Get samples from pipe
                if self.frame_pipe_out.poll(1.0 / 1000.0):
                    samples = self.frame_pipe_out.recv().reshape(-1, 1).astype("float32")
                    samples = resampler.process(samples, sample_multiplier)
                else:
                    continue
                while buffer_out.write_available < samples.shape[0] and play_action in stream.actions:
                    time.sleep(1.0 / 1000.0)
                if play_action not in stream.actions:
                    raise RuntimeError('Output ringbuffer underflow')
                buffer_out.write(samples)
        
    def start_processing(self, recurse = True):
        """
        Starts the streaming process.
        """
        if self.stop_process.value == True:
            self.stop_process.value = False
            self.audio_process = multiprocessing.Process(target = self.audio_runner)
            self.audio_process.start()
            self.audio_process = None # weakref pickle fix for new python versions
        super(RTMixerAudioSink, self).start_processing(recurse)
        
    def stop_processing(self, recurse = True):
        """
        Stops the streaming process.
        """
        super(RTMixerAudioSink, self).stop_processing(recurse)
        self.stop_process.value = True
