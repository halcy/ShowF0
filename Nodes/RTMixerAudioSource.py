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

class RTMixerAudioSource(Node.Node):
    def __init__(self, device_name_filter = None, sample_rate_out = 16000, block_size = 1024, exclusive_mode = False, name = "RTMixerAudioSource"):
        super(RTMixerAudioSource, self).__init__(has_inputs = False, name = name)

        # Parameters
        self.frame_pipe_out, self.frame_pipe_in = multiprocessing.Pipe(False)
        self.device_name_filter = device_name_filter
        self.sample_rate = sample_rate_out
        self.block_size = block_size

        # Exclusive mode setting
        self.exclusive_mode = exclusive_mode

        # Wind-down signaling
        self.stop_process = multiprocessing.Value('b', True)
  
    def find_device_and_rate(self):
        # Exclusive mode?
        exclusive = None
        if self.exclusive_mode:
            exclusive = sd.WasapiSettings(exclusive=True)

        # Figure out WASAPI id
        wasapi_id = None
        wasapi_default_device = None
        for id, api in enumerate(sd.query_hostapis()):
            if "wasapi" in api["name"].lower():
                wasapi_id = id
                wasapi_default_device = api["default_input_device"]
                break

        # If device name filter is set, find device. Otherwise, use default.
        device_id = None
        if self.device_name_filter is None:
            device_id = wasapi_default_device
        else:
            for id, device in enumerate(sd.query_devices()):
                if not device["hostapi"] == wasapi_id or device["max_input_channels"] <= 0:
                    continue
                if self.device_name_filter.lower() in device["name"].lower():
                    device_id = id
                    break
        
        # See if we can get the desired sample rate directly and without resampling
        device_sample_rate = self.sample_rate
        try:
            sd.check_input_settings(
                device = device_id, 
                channels = 1, 
                dtype = 'float32', 
                extra_settings = exclusive, 
                samplerate = device_sample_rate
            )
        except:
            # Doesn't work, have to resample
            device_sample_rate = sd.query_devices(device_id)["default_samplerate"]
        return device_id, device_sample_rate, exclusive
    
    def audio_runner(self):
        """Thread for getting data from the microphone"""
        # Get device and rate
        device, device_sample_rate, exclusive = self.find_device_and_rate()
        sample_multiplier = self.sample_rate / device_sample_rate
        
        # Open stream
        print("[Source] Opening device: ", device, "@", device_sample_rate)
        stream = rtmixer.Recorder(
            device = device,
            channels = 1, 
            blocksize = 0, 
            samplerate = device_sample_rate,
            latency = 0,
            extra_settings = exclusive
        )
        assert stream.dtype == 'float32'
        assert stream.samplesize == 4
        print("[Source] Latency: ", stream.latency, "+ buffer", round((self.block_size / self.sample_rate), 4))

        # Record
        buffer_in = rtmixer.RingBuffer(4, 2 ** math.ceil(math.log2(device_sample_rate)))
        record_action = stream.record_ringbuffer(buffer_in, allow_belated = True)
        device_block_size = int(self.block_size / sample_multiplier)
        read_buffer = np.zeros((device_block_size, 1), dtype='float32')
        with stream:
            while self.stop_process.value == False:
                while buffer_in.read_available < device_block_size and record_action in stream.actions:
                    sd.sleep(1)
                if record_action not in stream.actions:
                    raise RuntimeError('Input ringbuffer overflow')
                buffer_in.readinto(read_buffer)
                self.frame_pipe_in.send(read_buffer)
        
    def processing_runner(self):
        """Thread for getting data into the handlers."""
        # Do we need resampling?
        _, device_sample_rate, _ = self.find_device_and_rate()
        sample_multiplier = self.sample_rate / device_sample_rate
        print("[Source] Resampling using sinc_fastest and factor", round(sample_multiplier, 2))
        resampler = sr.Resampler('sinc_fastest', channels = 1)

        while self.stop_process.value == False:
            if self.frame_pipe_out.poll(1.0 / 1000.0):
                samples = self.frame_pipe_out.recv()
                samples = resampler.process(samples, sample_multiplier) # Consider moving this to audio thread?
            else:
                continue
            self.output_data(samples)

    def start_processing(self, recurse = True):
        """
        Starts the streaming process.
        """
        if self.stop_process.value == True:
            self.stop_process.value = False
            self.audio_process = multiprocessing.Process(target = self.audio_runner)
            self.audio_process.start()
            self.audio_process = None # weakref pickle fix for new python versions
            self.processing_process = multiprocessing.Process(target = self.processing_runner)
            self.processing_process.start()
            self.processing_process = None # weakref pickle fix for new python versions
        super(RTMixerAudioSource, self).start_processing(recurse)
        
    def stop_processing(self, recurse = True):
        """
        Stops the streaming process.
        """
        super(RTMixerAudioSource, self).stop_processing(recurse)
        self.stop_process.value = True
