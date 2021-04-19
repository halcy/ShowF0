import sys
import traceback
import math
import time
import multiprocessing
import socket
import json

import numpy as np

import time
from . import Node

##
# Enums
##
sampling_frequencies = {
    10240: "11", 
    5120: "10", 
    2048: "01", 
    512: "00"
}
input_sets = {
    "all": "11", 
    "i1-6_mi1-3": "10", 
    "i1-4_mi1-2": "01", 
    "i1-2_mi1": "00",
}
input_set_value_counts = {
    "all": 408,
    "i1-6_mi1-3": 312,
    "i1-4_mi1-2": 216,
    "i1-2_mi1": 120,
}
analog_out_gains = {
    16: "11", 
    4: "10", 
    2: "01", 
    1: "00",
}
analog_out_sources = {
    "aux_in": "1100",
    "mi4": "1011", 
    "mi3": "1010",
    "mi2": "1001",
    "mi1": "1000",
    "in8": "0111",
    "in7": "0110",
    "in6": "0101",
    "in5": "0100",
    "in4": "0011",
    "in3": "0010",
    "in2": "0001",
    "in1": "0000",
}
analog_out_max_channel = {
    "aux_in": 16,
    "mi4": 64, 
    "mi3": 64,
    "mi2": 64,
    "mi1": 64,
    "in8": 16,
    "in7": 16,
    "in6": 16,
    "in5": 16,
    "in4": 16,
    "in3": 16,
    "in2": 16,
    "in1": 16,
}
high_pass_freqs = {
    200.0: "11",
    100.0: "10",
    10.0: "01",
    0.3: "00",
}
low_pass_freqs = {
    4400.0: "11",
    900.0: "10",
    500.0: "01",
    130.0: "00",
}
modes = {
    "bipolar": "10",
    "differential": "01",
    "monopolar": "00"
}
input_ids = {
    "in1": 0,
    "in2": 1,
    "in3": 2,
    "in4": 3,
    "in5": 4,
    "in6": 5,
    "in7": 6,
    "in8": 7,
    "mi1": 8,
    "mi2": 9,
    "mi3": 10,
    "mi4": 11,
}
input_offsets = {
    "in1": (0 * 16, "none"),
    "in2": (1 * 16, "none"),
    "in3": (2 * 16, "none"),
    "in4": (3 * 16, "none"),
    "in5": (4 * 16, "none"),
    "in6": (5 * 16, "none"),
    "in7": (6 * 16, "none"),
    "in8": (7 * 16, "none"),
    "post_in": (8 * 16, "none"),
    
    "mi1": (0 * 64, "mi"),
    "mi2": (1 * 64, "mi"),
    "mi3": (2 * 64, "mi"),
    "mi4": (3 * 64, "mi"),
    "post_mi": (4 * 64, "mi"),
    
    "aux_in": (0, "rest"),
    "sampcount": (16, "rest"),
    "trigger": (17, "rest"),
    "buffer": (19, "rest"),
}
input_set_offsets = {
    "all": {
        "none": 0,
        "mi": input_offsets["post_in"][0],
        "rest": input_offsets["post_in"][0] + input_offsets["post_mi"][0],
    },
    "i1-6_mi1-3": {
        "none": 0,
        "mi": input_offsets["in7"][0],
        "rest": input_offsets["in7"][0] + input_offsets["mi4"][0],
    },
    "i1-4_mi1-2": {
        "none": 0,
        "mi": input_offsets["in5"][0],
        "rest": input_offsets["in5"][0] + input_offsets["mi3"][0],
    },
    "i1-2_mi1": {
        "none": 0,
        "mi": input_offsets["in3"][0],
        "rest": input_offsets["in3"][0] + input_offsets["mi2"][0],
    },
}
input_sizes = {
    "in1": 16,
    "in2": 16,
    "in3": 16,
    "in4": 16,
    "in5": 16,
    "in6": 16,
    "in7": 16,
    "in8": 16,
    
    "mi1": 64,
    "mi2": 64,
    "mi3": 64,
    "mi4": 64,
    
    "aux_in": 16,
    "sampcount": 1,
    "trigger": 1,
    "buffer":1,
}

##
# Package scoped private helpers
# All of these generate part of the config bytes
##
def _gen_bin_int(s):
    return str(s) if s<=1 else _gen_bin_int(s>>1) + str(s&1)

def _bin_str_to_bytes(binary_str):
    return bytearray([int(binary_str[i:i+8], 2) for i in range(0, len(binary_str), 8)])

def _gen_acq_set(decimator_active, record_on, fsamp_selector, channel_selector, acquisition_on):
    try:
        fsamp_selector = int(fsamp_selector)
    except:
        raise ValueError("Invalid sampling frequency")
        
    if not fsamp_selector in sampling_frequencies:
        raise ValueError("Invalid sampling frequency")
        
    if not channel_selector in input_sets:
        raise ValueError("Invalid input set")
    
    bit_str = "1"
    
    if decimator_active:
        bit_str += "1"
    else:
        bit_str += "0"
    
    if record_on:
        bit_str += "1"
    else:
        bit_str += "0" 
        
    bit_str += sampling_frequencies[fsamp_selector]
    bit_str += input_sets[channel_selector]
    
    if acquisition_on:
        bit_str += "1"
    else:
        bit_str += "0" 
    
    return bit_str

def _gen_an_out_config(anout_gain, anout_insel, anout_channel):
    if not anout_gain in analog_out_gains:
        raise ValueError("Invalid analog out gain")
        
    if not anout_insel in analog_out_sources:
        raise ValueError("Invalid analog out source")
    
    if anout_channel < 0 or anout_channel >= analog_out_max_channel[anout_insel]:
        raise ValueError("Invalid analog out channel")
    
    bit_str = "00"
    bit_str += analog_out_gains[anout_gain]
    bit_str += analog_out_sources[anout_insel]
    
    bit_str += "00"
    bit_str += _gen_bin_int(anout_channel).zfill(6)
    
    return bit_str

def _gen_in_config(hp_freq, lp_freq, mode):
    try:
        hp_freq = float(hp_freq)
    except:
        raise ValueError("Invalid high-pass frequency")
        
    try:
        lp_freq = float(lp_freq)
    except:
        raise ValueError("Invalid low-pass frequency")
    
    if not hp_freq in high_pass_freqs:
        raise ValueError("Invalid high-pass frequency")
    
    if not lp_freq in low_pass_freqs:
        raise ValueError("Invalid low-pass frequency")
    
    if not mode in modes:
        raise ValueError("Invalid mode")
        
    bit_str = ""
    bit_str += "00000000" # Muscle: "not defined"
    bit_str += "000000" # Sensor: "not defined"
    bit_str += "00" # Adapter: "not defined"
    bit_str += "00" # Side: "not defined"
    
    bit_str += high_pass_freqs[hp_freq]
    bit_str += low_pass_freqs[lp_freq]
    bit_str += modes[mode]
    
    return bit_str

def crc8(incoming):
    bin_bytes = _bin_str_to_bytes(incoming)
    check = 0
    for i in bin_bytes:
        check = _crc8_byte(i, check)
    return _gen_bin_int(check).zfill(8)

def _crc8_byte(b, crc):
    b2 = b
    if (b < 0):
        b2 = b + 256
    for i in range(8):
        odd = ((b2^crc) & 1) == 1
        crc >>= 1
        b2 >>= 1
        if (odd):
            crc ^= 0x8C
    return crc

def _int_to_millivolt(int_values):
    """
    Quattrocento-Specific: Conversion from raw integers to millivolts
    """
    return int_values * ((5.0 / math.pow(2.0, 16.0)) / 150.0) * 1000.0
    
class QuattrocentoDataSource(Node.Node):
    """
    Connects to an OT Quattrocento device and distributes frames to registered
    data sinks.
    """
    
    ##
    # Private helpers
    ##  
    def _gen_config(self, trigger_active, acquire):
        """
        Generate the config bytes for the values that are currently set
        """
        config = _gen_acq_set(
            self.acquisition_config["use_decimator"], 
            trigger_active, 
            self.acquisition_config["sampling_frequency"], 
            self.acquisition_config["input_set"], 
            acquire
        )
        config += _gen_an_out_config(
            self.anout_config["gain"],
            self.anout_config["source"], 
            self.anout_config["source_channel"], 
        )
        
        for i in range(12):
            config += _gen_in_config(
                self.input_configs[i]["hp_filter_at"], 
                self.input_configs[i]["lp_filter_at"], 
                self.input_configs[i]["derivation_mode"], 
            )
        config += crc8(config)
        return _bin_str_to_bytes(config)
    
    def _read_data(self, num_values):
        """
        Read some number of values from the input socket
        """
        data = b""
        while (len(data) < 2 * num_values) and (self.stop_process.value == False):
            try:
                data += self.socket.recv(2 * num_values - len(data))
            except socket.timeout:
                pass
        return np.frombuffer(data, dtype = 'int16').astype('float')
    
    ##
    # Public interface
    ##
    @staticmethod
    def create_from_json(configStr, connect_immediately = True, name = "QuattrocentoDataSource"):
        """Constructor replacement, return an instance configured according to string given"""
        config = json.loads(configStr)
        qc_source_obj = QuattrocentoDataSource(config["address"], config["port"], config["raw_data"], connect_immediately = connect_immediately)
        qc_source_obj.acquisition_config = config["acquisition_config"]
        qc_source_obj.anout_config = config["anout_config"]
        qc_source_obj.input_configs = config["input_configs"]
        qc_source_obj.channel_set = config["channel_set"]
        qc_source_obj.values_per_sample = config["values_per_sample"]
        qc_source_obj.validate_config()
        return qc_source_obj
    
    def __init__(self, quattrocento_addr, quattrocento_port = 23456, raw_data = False, connect_immediately = True, block_size = 8, name = "QuattrocentoDataSource", skip_startup=1024):
        """Sets up configuration and connect to the amplifier."""
        super(QuattrocentoDataSource, self).__init__(has_inputs = False, name = name)
        
        self.quattrocento_addr = quattrocento_addr
        self.quattrocento_port = quattrocento_port
        self.raw_data = raw_data
        self.block_size = block_size
        self.skip_startup = skip_startup
        
        self.trigger_active = False
        self.is_acquiring = multiprocessing.Value('b', False)
        self.frame_pipe_out, self.frame_pipe_in = multiprocessing.Pipe(False)
        
        self.acquisition_config = {
            "use_decimator": True,
            "sampling_frequency": 10240,
            "input_set": "all"
        }
        self.values_per_sample = input_set_value_counts["all"]
        self.anout_config = {
            "gain": 1,
            "source": "aux_in",
            "source_channel": 0
        }
        self.input_configs = []
        self.channel_set = None
        for i in range(12):
            self.input_configs.append({
                "hp_filter_at": 0.3, 
                "lp_filter_at": 4400, 
                "derivation_mode": "differential"
            })
    
        # Wind-down value
        self.stop_process = multiprocessing.Value('b', True)
    
        if connect_immediately == True:
            # Connect. If if fails, whatever, raise exception and let user deal with it.
            self.connect()
            
    def __del__(self):
        """Just closes the connection"""
        try:
            self.socket.settimeout(10.0)
            self.socket.close()
        except:
            pass
    
    def connect(self):
        """Connects to the amplifier.
           
           Note: You can't disconnect. Just delete the object and make a new
           one instead."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.quattrocento_addr, self.quattrocento_port))
        self.socket.sendall(self._gen_config(False, False))
    
    def get_json_config(self):
        """Get a nice-looking json formatted configuration"""
        config = {}
        config["address"] = self.quattrocento_addr
        config["port"] = self.quattrocento_port
        config["raw_data"] = self.raw_data
        config["acquisition_config"] = self.acquisition_config
        config["anout_config"] = self.anout_config
        config["input_configs"] = self.input_configs
        config["channel_set"] = self.channel_set
        config["values_per_sample"] = self.values_per_sample
        return json.dumps(config, indent = 4)
    
    def validate_config(self):
        """Throw an exception if the config is obviously broken."""
        self._gen_config(False, False)
    
    @staticmethod
    def get_input_channel_list(input_set, input_id):
        """Get list of indices corresponding to given input
           using the given input set."""
        firstChannel = input_offsets[input_id][0] + input_set_offsets[input_set][input_offsets[input_id][1]]
        lastChannel = firstChannel + input_sizes[input_id]
        return list(range(firstChannel, lastChannel))
            
    def set_acquisition_config(self, use_decimator, sampling_frequency, input_set, channel_set = None):
        """Sets the acquisition configuration. Cannot be used while acquiring.
           Also sets the channel set that is applied immediately upon recv-ing."""
        if self.is_acquiring.value:
            raise RuntimeError("Cannot change configuration while acquiring")
        
        self.acquisition_config = {
            "use_decimator": use_decimator,
            "sampling_frequency": sampling_frequency,
            "input_set": input_set
        }
        self._gen_config(False, False)
        self.values_per_sample = input_set_value_counts[input_set]
        self.channel_set = channel_set
        
    def set_input_config(self, input_id, hp_filter_at, lp_filter_at, derivation_mode):
        """Sets the input configuration. Cannot be used while acquiring."""
        if self.is_acquiring.value:
            raise RuntimeError("Cannot change configuration while acquiring")
        
        if not input_id in input_ids:
            raise ValueError("Invalid input id")
        
        input_idx = input_ids[input_id]
        self.input_configs[input_idx] = {
            "hp_filter_at": hp_filter_at,
            "lp_filter_at": lp_filter_at, 
            "derivation_mode": derivation_mode
        }
        self._gen_config(False, False)
    
    def set_analog_out_config(self, gain, source, source_channel):
        """Sets the analog output configuration. Cannot be used while acquiring."""
        if self.is_acquiring.value:
            raise RuntimeError("Cannot change configuration while acquiring")
        
        self.acquisition_config = {
            "gain": gain,
            "source": source,
            "source_channel": source_channel
        }
        self._gen_config(False, False)
    
    def set_trigger(self, trigger_active):
        """Pulls the trigger high or low"""
        self.trigger_active = trigger_active
        config = self._gen_config(self.trigger_active, self.is_acquiring)
        self.socket.sendall(config)
    
    def start_processing(self, recurse = True):
        """
        Instructs the quattrocento to acquire and send data.
        """
        # This ones processes never die
        if self.stop_process.value == True:
            self.stop_process.value = False
            multiprocessing.Process(target = self.processing_thread_runner, daemon=True).start()
            multiprocessing.Process(target = self.recording_thread_runner, daemon=True).start()     
        
        config = self._gen_config(self.trigger_active, True)
        self.socket.sendall(config)
        self.is_acquiring.value = True
        super(QuattrocentoDataSource, self).start_processing(recurse)
        
    def stop_processing(self, recurse = True):
        """
        nstructs the quattrocento to stop sending data.
        """
        super(QuattrocentoDataSource, self).stop_processing(recurse)
        config = self._gen_config(self.trigger_active, False)
        self.socket.sendall(config)
        self.is_acquiring.value = False
    
    def recording_thread_runner(self):
        """Thread for getting data from the device"""
        self.socket.settimeout(1.0 / 1024.0)
        while self.stop_process.value == False:
            emg_sample = self._read_data(self.values_per_sample * self.block_size)
            if self.stop_process.value == True:
                break
            
            emg_sample = emg_sample.reshape(self.block_size, self.values_per_sample)
            
            if not self.channel_set is None:
                emg_sample = emg_sample[:, self.channel_set]
            
            if self.raw_data == False:
                emg_sample = _int_to_millivolt(emg_sample)
            
            # Send new sample
            if self.skip_startup <= 0:
                if self.is_acquiring.value == True:
                    self.frame_pipe_in.send(emg_sample)
            else:
                self.skip_startup -= 1
        return
    
    def processing_thread_runner(self):
        """Thread for getting data into the handlers."""
        while self.stop_process.value == False:
            if self.frame_pipe_out.poll(1.0 / 1024.0):
                emg_sample = self.frame_pipe_out.recv()
                emg_sample = emg_sample.reshape(emg_sample.shape[0], emg_sample.shape[1])
            else:
                continue
            
            self.output_data(emg_sample)
        return
