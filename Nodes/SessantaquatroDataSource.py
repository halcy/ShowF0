import sys
import traceback
import math
import time
import multiprocessing
import socket
import json
import numpy as np

from . import Node

##
# Enums
##
sampling_frequencies = {
    4000: "11", 
    2000: "10", 
    1000: "01", 
    500: "00"
}
input_sets = {
    "64ch2aux": "11", 
    "32ch2aux": "10", 
    "16ch2aux": "01", 
    "8ch2aux": "00",
}
input_set_value_counts = {
    "64ch2aux": 34,
    "32ch2aux": 18,
    "16ch2aux": 10,
    "8ch2aux": 6,
}
input_modes = {
    "test": "111",
    "impedance": "110",
    "chained_diff": "010", # For groups of 32 electrodes - ch32 and 64 are monopolar
    "bipolar_adapter": "001"
    "monopolar": "000",
}

##
# Package scoped private helpers
##
def _gen_bin_int(s):
    return str(s) if s<=1 else _gen_bin_int(s>>1) + str(s&1)

def _bin_str_to_bytes(binary_str):
    return bytearray([int(binary_str[i:i+8], 2) for i in range(0, len(binary_str), 8)])

def _int_to_millivolt(int_values): # TODO this is wrong unless the Sessantaquatro uses a 150 V/V amplicication like the Quattrocento does
    return int_values * ((5.0 / math.pow(2.0, 16.0)) / 150.0) * 1000.0
    
class SessantaquatroDataSource():
    """
    Connects to an OT Sessantaquatro device and distributes frames to registered
    data sinks.
    """
    def _read_data(self, num_values):
        """
        Reads some 16 bit signed integers from the socket
        """
        data = b""
        while len(data) < 2 * num_values and RunManager.RunManager.running():
            try:
                data += self.socket.recv(2 * num_values - len(data))
            except socket.timeout:
                pass
        return np.frombuffer(data, dtype = 'int16').astype('float')
    
   def __init__(self, address, sample_rate = 1000, input_set = "32ch2aux", input_mode = "chained_diff", name = "SessantaquatroDataSource"):
        """
        Initialize a SessantaquatroDataSource.
        """
        super(SessantaquatroDataSource, self).__init__(has_inputs = False, name = name)
        self.address = address
        self.sample_rate = sample_rate
        self.input_set = input_set
        self.input_mode = input_mode
        
        self.frame_pipe_out, self.frame_pipe_in = multiprocessing.Pipe(False)
        self.read_process = None
        self.processing_process = None
        
        self.socket = None
        
    def __gen_config(self, close_connection = False):
        config_str = ""
        
        # CONFIG BYTE 0
        config_str += "0" # What we are sending is new settings
        config_str += sampling_frequencies[self.sample_rate]
        config_str += input_sets[self.input_set]
        config_str += input_modes[self.input_mode]
        
        # CONFIG BYTE 1
        config_str += "0" # "Low-Res" sampling (high res does not work yet according to docs)
        config_str += "1" # Use high-pass filter at (sampling frequency / 190) Hz
        config_str += "00" # Standard input range (this should be okay for EMG - to be verified)
        config_str += "00" # Network control of data transfer
        config_str += "0" # Do not record to microsd
        
        # Send data or no? 1 means "send me data", 0 means "stop doing that and immediately close the TCP socket"
        if close_connection == True:
            config_str += "0"
        else:
            config_str += "1"
        
        # That should be all we strictly HAVE to send, skip the rest
        return _bin_str_to_bytes(config_str)
    
    def read_process(self):
        """
        Streams the data from the device
        """
        # Sending is go
        values_per_sample = input_set_value_counts[self.input_set]
        config = self.__gen_config(True)
        self.socket.sendall(config)
        
        self.socket.settimeout(1.0 / 1024.0)
        while True:
            emg_sample = self._read_data(self.values_per_sample * 8) # Why 8 tho, like why 8 frames at a time. does this make sense with 1000Hz?
            emg_sample = emg_sample.reshape(8, self.values_per_sample)
            
            # TODO
            #if self.rawData == False:
            #    emg_sample = _intToMillivolt(emg_sample)
            
            # Send new sample
            self.frame_pipe_in.send(emg_sample)
        return
    
    def processing_process(self):
        """
        Calls frame callbacks for each frame.
        """
        while True:
            # Grab sample from pipe
            if self.frame_pipe_out.poll(1.0 / 1024.0):
                emg_sample = self.frame_pipe_out.recv()
                # emg_sample = emg_sample.reshape(emg_sample.shape[0], emg_sample.shape[1]) # IDK why needed? should not be
            else:
                continue
            
            # Process sample
            self.output_data(emg_sample)
        return
    
    def start_processing(self, recurse = True):
        """
        Connects and starts the process.
        """
        if self.socket == None:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.address, 23456)) # TODO port correct?
        
        if self.read_process is None:
            self.read_process = multiprocessing.Process(target = self.read_process)
            self.read_process.start()
            
        if self.processing_process is None:
            self.processing_process = multiprocessing.Process(target = self.processing_process)
            self.processing_process.start()
            
        super(Sender, self).start_processing(recurse)
        
    def stop_processing(self, recurse = True):
        """
        Stops the streaming process.
        """
        super(Sender, self).stop_processing(recurse)
        
        # Stop processes FIRST
        if not self.read_process is None:
            self.read_process.terminate()
        self.read_process = None
        
        if not self.processing_process is None:
            self.processing_process.terminate()
        self.processing_process = None
        
        # THEN close the socket
        try:
            # First, try a proper teardown
            config = self.__gen_config(False)
            self.socket.sendall(config)
        
            self.socket.settimeout(10.0)
            self.socket.close()
        except:
            pass
