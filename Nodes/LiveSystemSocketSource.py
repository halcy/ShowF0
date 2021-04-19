import sys
import traceback
import math
import time
import queue
import multiprocessing
import socket

import numpy as np

from itertools import *

from . import TimeSeriesLogger
from . import RunManager

class LiveSystemSocketSource():
    """
    Connects to a socket and returns the received data as a numpy array.
    Input data is assumed to be 16 bit signed little endian integers.
    """
    def __init__(self, socketHost, socketPort = 17389, channelCount = 2):
        """Just saves the connection data - does not connect or start 
           recording."""
        self.socketHost = socketHost
        self.socketPort = socketPort
        
        self.channelCount = channelCount
        
        self.framePipeOut, self.framePipeIn = multiprocessing.Pipe(False)
        
        self.frameCallbacks = []

    def addFrameCallback(self, frameCallback):
        """Adds a function to call when a frame is ready.
           This function is called with the newly ready frame as a numpy 
           ndarray, time in rows, channels in columns."""
        self.frameCallbacks.append(frameCallback)

    def recordingThreadRunner(self):
        """Thread for getting data from the socket."""
        try:
            inputSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            inputSocket.connect((self.socketHost, self.socketPort))
            sampleCount = 0
            while RunManager.RunManager.running():
                sampleCount += 1
                try:
                    # Grab sample
                    sample = np.empty((1, self.channelCount), dtype = np.int16)
                    readData = 0
                    while readData < self.channelCount * 2 and RunManager.RunManager.running():
                        readData += inputSocket.recv_into(sample[:,readData:], self.channelCount * 2)
                        
                    # Enqueue new sample
                    TimeSeriesLogger.TimeSeriesLogger.logPoint("SocketSourceGetData", sample)
                    self.framePipeIn.send(sample)
                except KeyboardInterrupt:
                    print("Received interrupt (SocketSource Recording), winding down.")
                    RunManager.RunManager.windDown()
        except:
            print("Error (SocketSource Recording), winding down.")
            RunManager.RunManager.windDown()
            
        inputSocket.close()
        time.sleep(0.5)
        return
    
    def processingThreadRunner(self):
        """Thread for getting data into the handlers."""
        try:
            while RunManager.RunManager.running():
                sample = self.framePipeOut.recv()
                
                # Process sample
                #print(str(self) + " returns frame at " + ("%.6f" % float(time.time())))
                for frameCallback in self.frameCallbacks:
                    frameCallback(sample)
            
        except KeyboardInterrupt:
            print("Received interrupt (SocketSource Processing), winding down.")
            RunManager.RunManager.windDown()
            
        TimeSeriesLogger.TimeSeriesLogger.writeLogs()
        return
        
    def startRecording(self):
        """Start up worker threads that can begin to collect samples.
           As both threads hold a reference to this object, it should never
           go out of scope - until the threads are wound down."""

        # Spin-wait for empty processing queue
        while self.framePipeOut.poll():
            self.framePipeOut.recv()
            
        # Begin processing
        multiprocessing.Process(target = self.processingThreadRunner).start()
        multiprocessing.Process(target = self.recordingThreadRunner).start()
    
