import numpy as np
from itertools import islice

import time
import multiprocessing

from . import Node

class UtteranceCutter(Node.Node):
    """
    Takes streams of data and cuts them up, optionally performing marker synchronization.
    
    Doesn't have outputs - instead, returns data since the last "begin_utterance" when
    "cut_utterance" is called, with the given padding.
    """
    def __init__(self, padding_ms = 1, name = "UtteranceCutter", quiet = False):
        """Initializes the cutter for operating with the given padding."""
        super(UtteranceCutter, self).__init__(name = name)
        self.padding_ms = padding_ms
        self.frame_buffers = []
        self.sample_rates = []
        self.pipes = []
        self.cutting_process = None
        self.run_cutting = multiprocessing.Value('b', False)
        self.signal_queue = multiprocessing.Queue()
        self.utterance_queue = multiprocessing.Queue()
        
        self.cut_marks = []
        self.planned_cuts = []
        self.marker_detectors = []
        self.marker_delays_ms = []
        self.quiet = quiet
        
    def cutting(self):
        """
        Process for ready frames. Stores frames in their respective buffers and,
        soon as enough data is available, pops them off."""
        while self.run_cutting.value:
            # Get new data
            for idx, pipe in enumerate(self.pipes):
                if pipe[0].poll(1.0 / 20000.0):
                    self.frame_buffers[idx].extend(pipe[0].recv())
            
            # Got signalled?
            if not self.signal_queue.empty():
                command = self.signal_queue.get()

                # Mark cut start
                if command == "start":
                    for (idx, frame_buffer) in enumerate(self.frame_buffers):
                        self.cut_marks[idx] = len(frame_buffer)
                        
                # Mark end cut and prepare to return data
                if command == "cut":
                    cut_info = []
                    for (idx, frame_buffer) in enumerate(self.frame_buffers):
                        cut_start = int(max(0, self.cut_marks[idx] - (self.padding_ms / 1000.0)  * self.sample_rates[idx]))
                        cut_end = int(len(frame_buffer) + (self.padding_ms / 1000.0) * self.sample_rates[idx])
                        cut_info.append((cut_start, cut_end))
                    self.planned_cuts.append(cut_info)
                
                if command == "clear":
                    for idx in range(len(self.frame_buffers)):
                        self.frame_buffers[idx] = []
                        
            # Output if data is there
            planned_cuts_new = []
            for cut_info in self.planned_cuts:
                # Check if we have enough data
                can_cut = True
                for (list_cut_info, frame_buffer) in zip(cut_info, self.frame_buffers):
                    if len(frame_buffer) < list_cut_info[1]:
                        can_cut = False
                    
                # If so, cut, potentially factoring in marker-inferred signal delays
                if can_cut:
                    utterance = []
                    for (list_cut_info, frame_buffer) in zip(cut_info, self.frame_buffers):
                        utterance.append(np.array(frame_buffer[list_cut_info[0]:list_cut_info[1]]))
                    
                    if len(self.marker_detectors) > 0:
                        # Find markers
                        marker_positions_sec = []
                        earliest_marker_sec = float("inf")
                        for(utterance_frames, marker_detector, sample_rate, marker_delay_ms) in zip(utterance, self.marker_detectors, self.sample_rates, self.marker_delays_ms):
                            for (frame_idx, frame) in enumerate(utterance_frames):
                                if marker_detector(frame):
                                    marker_positions_sec.append(frame_idx / float(sample_rate) + marker_delay_ms / 1000.0)
                                    if not self.quiet:
                                        print("Marker found:", frame_idx, frame_idx / float(sample_rate), marker_delay_ms / 1000.0, frame_idx / float(sample_rate) + marker_delay_ms / 1000.0)
                                    earliest_marker_sec = min(marker_positions_sec[-1], earliest_marker_sec)
                                    break
                                    
                        # No marker? Throw.
                        if len(marker_positions_sec) < len(self.marker_detectors):
                            raise AssertionError("Found no marker in utterance.")
                            
                        # Make smallest synchronizing cut
                        marker_adjusted_utterance = []
                        min_len_sec = 999999999
                        for(utterance_frames, marker_position_ms, sample_rate) in zip(utterance, marker_positions_sec, self.sample_rates):
                            marker_shift = (marker_position_ms - earliest_marker_sec) * float(sample_rate)
                            if not self.quiet:
                                print("Marker: ", sample_rate, marker_position_ms, marker_shift)
                            marker_adjusted_utterance.append(utterance_frames[int(marker_shift):,:])
                            min_len_sec = min(float(len(marker_adjusted_utterance[-1]) / sample_rate), min_len_sec)
                            
                        # Now, cut to same length
                        for idx, sample_rate in enumerate(self.sample_rates):
                            cut_end = int(min_len_sec * sample_rate)
                            marker_adjusted_utterance[idx] = marker_adjusted_utterance[idx][:cut_end, :]
                        utterance = marker_adjusted_utterance
                    
                    # Pass data back to cutUtterance
                    self.utterance_queue.put(utterance)
                else:
                    planned_cuts_new.append(cut_info)
                    
            self.planned_cuts = planned_cuts_new
            
    def add_data(self, data, data_id):
        """
        Adds data to be received in joining-process.
        """
        self.pipes[data_id][1].send(data)
        
    def set_inputs(self, input_classes):
        """
        Adds a source for frames to the cutter.
        
        It is possible to pass a list of tuples that can 
        optionally include sample rate (assumed "1" if not
        passed), marker-detector functions for cutting and
        additional marker delays (in milliseconds).
        """
        # Listify and call superclass setup
        if not isinstance(input_classes[0], (list, tuple)):
            input_classes = [[x] for x in input_classes]
        super(UtteranceCutter, self).set_inputs([x[0] for x in input_classes])
        
        # Set up
        for input_class in range(len(self.get_inputs())):
            self.frame_buffers.append([])
            self.pipe_out, self.pipe_in = multiprocessing.Pipe(False)
            self.pipes.append((self.pipe_out, self.pipe_in))
            self.cut_marks.append(0)
            
            # Optional parameter #1: Sample rates
            if len(input_classes[input_class]) > 1:
                self.sample_rates.append(input_classes[input_class][1])
            else:
                self.sample_rates.append(1)
            
            # Optional parameter #2: Marker detectors
            if len(input_classes[input_class]) > 2:
                self.marker_detectors.append(input_classes[input_class][2])
                
            # Optional parameter #3: Marker delays
            if len(input_classes[input_class]) > 3:
                self.marker_delays_ms.append(input_classes[input_class][3])
            else:
                self.marker_delays_ms.append(0)
                
    def start_processing(self,recurse = True):
        """
        Starts the cutting process
        """
        if self.cutting_process is None:
            self.cutting_process = multiprocessing.Process(target = self.cutting)
            self.run_cutting.value = True
            self.cutting_process.start()
            super(UtteranceCutter, self).start_processing(recurse)
        
    def stop_processing(self, recurse = True):
        """
        Stops the cutting process.
        """
        super(UtteranceCutter, self).stop_processing(recurse)
        if not self.cutting_process is None:
            self.cutting_process.terminate()
        self.cutting_process = None
        self.run_cutting.value = False     

    def begin_utterance(self):
        """ Tells the recording thread to update the start markers"""
        self.signal_queue.put("start")

    def cut_utterance(self):
        """Gets the data since the last start mark, with padding and marker
           synchronization as set up for this utterance cutter.
           
           Note that this function will block until enough data to
           actually return the utterance becomes available. Overlapping
           two cuts is possible in theory, but not recommended."""
        self.signal_queue.put("cut")
        utterance_data = self.utterance_queue.get()
        self.output_data(utterance_data)
        return utterance_data
    
    def clear_cutter(self):
        """Removes all data from the cut lists. It is not legal to call this function between
           begin_utterance and cut_utterance.
           
           Note that if clear is quickly followed by begin, the initial delay may be off."""
        self.signal_queue.put("clear")
