import pyaudio
import socket
import multiprocessing
import select
import numpy as np
import samplerate

"""
Simplest possible audio source.
Not configurable or whatever, just blasts stereo audio out at 16khz the end.
"""

from . import Node

class AudioSource(Node.Node):
    def __init__(self, device_name_filter = "USB", block_size=128, name = "AudioSource"):
        super(AudioSource, self).__init__(has_inputs = False, name = name)
        self.possible_rates = [16000, 32000, 48000, 96000, 128000]
        
        self.frame_pipe_out, self.frame_pipe_in = multiprocessing.Pipe(False)
        self.device_name_filter = device_name_filter
        self.sample_rate = None
        self.block_size = block_size

        # Wind-down value
        self.stop_process = multiprocessing.Value('b', True)
       
    def test_sample_rates(self, p, device_id):
        supported_rates = []
        for rate in self.possible_rates:
            is_supported = False
            try:
                is_supported = p.is_format_supported(
                    input_device = device_id,
                    input_format = pyaudio.paInt16, 
                    input_channels = 1,
                    rate = rate,
                )
            except:
                is_supported = False
            if is_supported:
                supported_rates.append(rate)
        return supported_rates
    
    def audio_runner(self):
        """Thread for getting data from the microphone"""

        # Find matching audio device
        p = pyaudio.PyAudio()
        self.api_id = None
        self.device_id = None
        num_apis = p.get_host_api_count()

        for j in range(0, num_apis):
            info = p.get_host_api_info_by_index(j)
            numdevices = info.get('deviceCount')
            for i in range(0, numdevices):
                if (p.get_device_info_by_host_api_device_index(j, i).get('maxInputChannels')) >= 1:
                    device_name = p.get_device_info_by_host_api_device_index(j, i).get('name')
                    if self.device_name_filter in device_name and len(self.test_sample_rates(p, i)) != 0:
                        self.api_id = j
                        self.device_id = i
                        print("Found device with id " + str(self.api_id) + "." + str(self.device_id) + ": " + device_name)
                        
        if self.device_id is None:
            print("No devices found that match filter. Device list:")
            for j in range(0, num_apis):
                info = p.get_host_api_info_by_index(j)
                numdevices = info.get('deviceCount')
                for i in range(0, numdevices):
                    dev_info = p.get_device_info_by_host_api_device_index(j, i)
                    device_name = dev_info.get('name')
                    supported_rates = self.test_sample_rates(p, i)
                    print("*", device_name, "channels:", dev_info.get('maxInputChannels'), "rates:", supported_rates)
            raise AssertionError("No device.")
        
        # Find best samplerate
        self.sample_rate = min(self.test_sample_rates(p, self.device_id))
        sample_multiplier = int(self.sample_rate / 16000)
        resampler = samplerate.Resampler('sinc_fastest', channels = 1)
        
        # Open stream
        stream = p.open(
            format = pyaudio.paInt16, 
            channels = 1,
            rate = self.sample_rate, 
            input = True,
            frames_per_buffer = self.block_size,
            input_device_index = self.device_id,
        )

        # Record
        while self.stop_process.value == False:
            data = stream.read(self.block_size, exception_on_overflow = False)
            samples = np.frombuffer(data, dtype = 'int16').astype('float').reshape(self.block_size, 1)
            samples_res = resampler.process(samples, sample_multiplier)
            self.frame_pipe_in.send(samples_res)
        p.terminate()
        
    def processing_runner(self):
        """Thread for getting data into the handlers."""
        while self.stop_process.value == False:
            if self.frame_pipe_out.poll(1.0 / 1024.0):
                samples = self.frame_pipe_out.recv()
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
            
        super(AudioSource, self).start_processing(recurse)
        
    def stop_processing(self, recurse = True):
        """
        Stops the streaming process.
        """
        super(AudioSource, self).stop_processing(recurse)
        self.stop_process.value = True
