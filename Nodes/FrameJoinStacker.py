import numpy as np
from collections import deque
from itertools import islice
import os

import time
from multiprocessing import Process, Pipe, Value

from . import Node

class FrameJoinStacker(Node.Node):
    """
    Takes a stream of frames and combines and stacks them.
    
    Set warm_start to True to pre-fill with zeros so frames start coming out immediately
    """
    def __init__(self, stacking_height = 1, warm_start = False, name = "FrameJoinStacker"):
        """Initializes a frame combiner / stacker with the given stacking height,
           stacking only into the past, inclusive current frame - i.e. a stacking 
           height of 15 will result in a 15 input frames long frame, and to get, for
           example, TD-15 features, a stacking height of 31 is required. Returned
           data is a list of features, containing whatever the frame sources put out,
           stacked to the stacking height."""
        super(FrameJoinStacker, self).__init__(name = name)
        self.stacking_height = stacking_height
        self.frame_buffers = []
        self.frame_delays = []
        self.pipes = []
        self.joining_process = None
        self.run_join = Value('b', False)
        self.warm_start = warm_start
        self.warm_start_done = True
        if self.warm_start == True:
            self.warm_start_done = False
            
    def joining(self):
        """
        Process for ready frames. Stores frames in their respective buffers and,
        soon as enough data is available, pops them off."""
        while self.run_join.value:
            # Pull one frame for each buffer 
            for i in range(0,len(self.pipes)):
                if not self.warm_start_done:
                    pipe_frame = self.pipes[i][0].recv()
                    for j in range(self.stacking_height - 1):
                        self.frame_buffers[i].append(np.zeros(pipe_frame.shape))
                    self.frame_buffers[i].append(pipe_frame)
                else:
                    self.frame_buffers[i].append(self.pipes[i][0].recv())
            self.warm_start_done = True
            
            # Check for frame delay, if delay phase is active, just bail and drop frame
            if self.frame_delays[i] != 0 and self.sync_streams == True:
                self.frame_delays[i] = self.frame_delays[data_id] - 1
                continue
        
            # Check all buffers sizes, bail if any are too small
            buffers_too_short = False
            for i in range(0, len(self.frame_buffers)):
                if len(self.frame_buffers[i]) < self.stacking_height:
                    buffers_too_short = True
                    break
            if buffers_too_short == True:
                continue

            # Have a complete frame, grab stacked frame and pop one datum off each buffer
            return_buffer = []
            for i in range(0, len(self.frame_buffers)):
                return_buffer.append(list(islice(self.frame_buffers[i], 0, self.stacking_height)))
                self.frame_buffers[i].popleft()
            self.output_data(return_buffer)

    def add_data(self, data, data_id):
        """
        Adds data to be received in joining-process.
        """
        if len(data.shape) == 1:
            data = data.reshape(-1, data.shape[0])
        for frame in data:
            self.pipes[data_id][1].send(frame)
        
    def set_inputs(self, input_classes, delay_frames = None):
        """Adds a source for frames to the join/stacker."""
        super(FrameJoinStacker, self).set_inputs(input_classes)
        
        for input_class in range(len(self.get_inputs())):
            self.frame_buffers.append(deque([]))
            self.pipe_out, self.pipe_in = Pipe(False)
            self.pipes.append((self.pipe_out, self.pipe_in))

            if delay_frames is None:
                self.frame_delays.append(0)
            else:
                self.frame_delays.append(delay_frames[i])
        
    def start_processing(self,recurse = True):
        """
        Starts the stacker process
        """
        if self.joining_process is None:
            self.joining_process = Process(target = self.joining)
            self.run_join.value = True
            self.joining_process.start()
            super(FrameJoinStacker, self).start_processing(recurse)
        
    def stop_processing(self, recurse = True):
        """
        Stops the stacker process.
        """
        super(FrameJoinStacker, self).stop_processing(recurse)
        if not self.joining_process is None:
            self.joining_process.terminate()
        self.joining_process = None
        self.run_join.value = False     
