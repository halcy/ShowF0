#!/usr/bin/env python3
"""
Simple Jack audio source
"""

import numpy as np
import threading
import multiprocessing
import logging
import time
import samplerate
import copy

from . import Node

class JackAudioSource(Node.Node):
    def __init__(self, sample_rate = 16000, block_size = 256, gain = 1.0, name = "JackAudioSource"):
        super(JackAudioSource, self).__init__(has_inputs = False, name = name)
        
        self.block_size = block_size
        self.sample_rate = sample_rate
        self.gain = gain
        
        # IPC setup
        self.output_process = None
        self.sample_pipe_out, self.sample_pipe_in = multiprocessing.Pipe(False)
        self.client_reset()
        
    def client_reset(self):
        import jack        
         # Jack setup
        self.client = jack.Client('JackAudioSource')
        self.client.blocksize = self.block_size
        self.is_active = False
        
        # Debug
        self.start_time = 0
        self.sample_count = 0
        self.xrun_count = 0
        
        self.ports = []
        self.ports.append(self.client.inports.register('audio_in_0'))
        self.ports.append(self.client.inports.register('audio_in_1'))

        # Callback setup
        self.client.set_process_callback(self.process)
        self.client.set_xrun_callback(self.xrun)
        
    def process(self, block_size):
        if self.is_active:
            self.sample_pipe_in.send(np.array([
                self.ports[0].get_array()[:],
                self.ports[1].get_array()[:],
            ]))
            
    def xrun(self, usecs):
        self.xrun_count += 1
        if self.xrun_count % 50 == 0:
            logging.info("xruns in: " + str(self.xrun_count) + ", samples: " + str(self.sample_count) + ", time: " + str(time.time() - self.start_time))
            
    def get_stats(self):
        return([self.pipe_fill.value, self.xrun_count, self.sample_count, time.time() - self.start_time])
    
    def process_samples(self):
        resampler = samplerate.Resampler('sinc_fastest', channels=2)
        resampler.process(np.zeros((2048, 2)), self.sample_rate / self.client.samplerate)
        while True:
            self.output_data(resampler.process(self.sample_pipe_out.recv().T, self.sample_rate / self.client.samplerate) * self.gain)
    
    def start_processing(self, recurse = True):
        """
        Start recording and stream data
        """
        if self.is_active == False:
            if self.output_process is None:
                self.output_process = multiprocessing.Process(target = self.process_samples)
                self.output_process.start()
            self.start_time = time.time()
            self.client.activate()
            
            # Connect input ports
            target_ports = self.client.get_ports(is_physical = True, is_output = True, is_audio = True)
            self.client.connect(target_ports[0], self.ports[0])
            self.client.connect(target_ports[1], self.ports[1])
            self.is_active = True
        super(JackAudioSource, self).start_processing(recurse)
        
    def stop_processing(self, recurse = True):
        super(JackAudioSource, self).stop_processing(recurse)
        self.client.deactivate()
        self.client.close()
        self.is_active = False
        self.client_reset()

        if not self.output_process is None:
            self.output_process.terminate()
        self.output_process = None
