import logging
import os
import sys
import errno
import random

import time
import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import PCA
import multiprocessing
import scipy
import gc

from . import Node 
from . import FrameJoinStacker
from . import LambdaNode
from . import FrameBuffer
from . import Receiver


class UnitSelector(Node.Node):
    """  
    """
    def __init__(self, unit_length, unit_shift,window_size,frame_shift, decode_shift = None,train_mode = True,name = "UnitSelector"):
        """ 
        """ 
        super(UnitSelector,self).__init__(name = name)
        

        # Set variables up
        self.unit_length = unit_length
        self.unit_shift = unit_shift
        self.window_size = window_size
        self.frame_shift = frame_shift
        if decode_shift is None:
           self.decode_shift = unit_shift
        else:
            self.decode_shift = decode_shift
        self.train_mode = train_mode
        
        self.manager = multiprocessing.Manager()
        self.lists = self.manager.list([])
        self.codebook = []
        self.codebook_source = []
        self.codebook_target = []
        self.feat_in_pipe_out, self.feat_in_pipe_in = multiprocessing.Pipe(False)
        self.mapping_process = None
        self.run_map = multiprocessing.Value('b', False)
        self.normed_source = []
        self.pca = None
        
        
        self.decode_buffer = FrameBuffer.FrameBuffer(self.unit_length,self.decode_shift, 1000, name = self.name + ".decode_buffer")
        self.pipe_add = LambdaNode.LambdaNode(lambda x : self.feat_in_pipe_in.send(x), name = "pipe_add")(self.decode_buffer)
        
        
        


        self.decoder = LambdaNode.LambdaNode(self.decode, name = name + ".decoder")
        self.reconstruct_buffer = FrameBuffer.FrameBuffer(self.unit_length,(self.window_size/self.frame_shift)/self.decode_shift, 1000,
                                                          warm_start = True, name = self.name + ".recon_buffer")(self.decoder)
        self.reconstructer = LambdaNode.LambdaNode(self.reconstruct_audio,
                                                   name = name + ".reconstructer")(self.reconstruct_buffer)

    def decode(self,data_signal):
        data_signal = self.pca.transform(data_signal)
        data_signal = np.array(data_signal).reshape(1,-1)
        
        audio_feats = self.codebook_target

        cos_sims = np.dot(data_signal,self.normed_source)
        max_score_idx = np.argmax(cos_sims)
        result = audio_feats[max_score_idx]
        return np.array(result.reshape(1,-1))

    
    def ensure_same_length(self):
        min_len = min(len(self.lists[0]), len(self.lists[1]))
        while len(self.lists[0]) > min_len:
            self.lists[0].pop()
        while len(self.lists[1]) > min_len:
            self.lists[1].pop()

    def reconstruct_audio(self, decode_result):
        frames = []
        weight = 0
        samples_per_unit = int((np.shape(decode_result)[1]/self.unit_length))
        
        decode_result = decode_result[::-1]
        for i,frame in enumerate(decode_result):
            #apply hamming window
            #window = scipy.signal.hamming(len(frame))
            #win_frame = np.multiply(window,frame)
            win_frame = frame
            
            recon_frame = win_frame[(i*samples_per_unit):((i+1)*samples_per_unit)]
            #weight += np.mean(window[(i*samples_per_unit):((i+1)*samples_per_unit)])
            
            
            frames.append(recon_frame)    
        result = np.mean(frames,axis=0,keepdims= True)
        #result = np.sum(frames,axis=0,keepdims= True)
        #result = np.array([x/weight for x in result])
        
        self.output_data(result)

    def set_train_mode(self,train_mode):
        """
        Sets the train mode to either train or decode mode.
        train_mode - True = train_mode , False = decode_mode.
        """
        self.train_mode = train_mode

    def add_data(self, sample, data_id):
        """
        Add a single frame of data

        If train_mode is True, it will be used to receive data for the code book.
        Else it is used to decode and passthrough data.
        """
        if self.train_mode is True:
            sample = np.array(sample).flatten()
            self.lists[data_id].append(sample)
        else:
            sample = np.array([sample])
            self.decode_buffer.add_data(sample)

    def set_inputs(self, input_classes):
        """Adds a source for frames to the list of inputs."""
        
        super(UnitSelector, self).set_inputs(input_classes)
            
        for input_class in range(len(self.get_inputs())):
            data_list = self.manager.list([])
            self.lists.append(data_list)
        

    
    def start_training(self):
        """
        Builds the codebook. 
        
        """ 
        
        #TODO: assert same length
        audio_feat = np.array([x for x in self.lists[1]]).astype("float32")
        ecog_feat = np.array([x for x in self.lists[0]]).astype("float32")
        print("ecog_feat",ecog_feat.shape)
        #del self.lists
        #pca
        pca = PCA(n_components = 0.7, svd_solver = 'full')
        #pca = PCA()
        pca.fit(ecog_feat)
        self.pca = pca
        ecog_feat = pca.transform(ecog_feat)
        print("pca",ecog_feat.shape)
        
        feat_in = []
        feat_out = []
        num_windows= 1+ int((len(audio_feat)-self.unit_length)/self.unit_shift)

        #create units with shift
        for i in range(num_windows):
            input_unit = ecog_feat[(i*self.unit_shift):(i*self.unit_shift+self.unit_length)]
            output_unit = audio_feat[(i*self.unit_shift):(i*self.unit_shift+self.unit_length)]
            feat_in.append(input_unit)
            feat_out.append(output_unit)
        
            
        del audio_feat
        del ecog_feat
        
        self.codebook_source = np.array([x.flatten() for x in feat_in]).astype("float32")
        self.codebook_target = np.array(feat_out).astype("float32")
        
        del feat_out
        
        
        source_norms = [norm(x.flatten()) for x in feat_in]
        self.normed_source = np.array([(ai)/bi for ai,bi in zip(self.codebook_source,source_norms)]).T.astype("float32")
        
        del self.codebook_source
        
        

        
        
    def mapping_runner(self): 
        """
        Decodes frames and calls callbacks.
        """


        while self.run_map.value:
            sample = self.feat_in_pipe_out.recv()
            self.decoder.add_data(sample,data_id = 0)
            
    
                
    def start_processing(self,recurse = True):
        """
        Starts the decode process. train_mode has to be False to work.
        """
        if self.train_mode == False:
            self.run_map.value = True
            mapping_process = multiprocessing.Process(target = self.mapping_runner)
            mapping_process.start()
        super().start_processing(recurse)
        
    def stop_processing(self, recurse = True):
        """
        Stops the streaming process.
        """
        super(Sender, self).stop_processing(recurse)
        self.run_map.value = False
        if not self.mapping_process is None:
            self.mapping_process.terminate()
        self.mapping_process = None
                
