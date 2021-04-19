import multiprocessing
import logging
import numpy as np
from . import Node

import matplotlib.pyplot as plt
from scipy.signal import hanning
import sys
sys.path.append('/share/documents/cherff/ECoG/Northwestern-IntraOP')
import MelFilterBank as mel


class SpawnNopNode(Node.Node):
    """
    This node is not applying a function (No-operation) on the data stream but simply spawns a
    new process and forwards the data stream over a pipe into the new process. It can be used
    for nodes which are a composite of other nodes and connected with the set_passthrough
    function to run in a separate process.
    """
    def __init__(self, block_size, reader, writer, name='SpawnNopNode'):
        """
        Setting up a new NOP node

        :param block_size: Samples are passed in batches (blocks) to the next node
        :param reader: End connection object of a Pipe object used to read the data
        :param writer: Start connection object of the same Pipe object used in the reader
        :param name: Name of the node
        """
        super(SpawnNopNode, self).__init__(name=name)
        self.logger = logging.getLogger('Node[{}]'.format(name))

        self.block_size = block_size
        self.reader = reader
        self.writer = writer

        self.conversion_process = None

    def pass_data_to_next_node(self, reader):
        while True:
            block = []
            while len(block) < self.block_size:
                data_frame = reader.recv()
                block.extend(data_frame)

            self.output_data(np.array(block).reshape(-1, 1))

    def add_data(self, data_frame, data_id=None):
        """
        Forwards the data stream into the Pipe for inter-process communication.
        """
        self.writer.send(data_frame)

    def start_processing(self, recurse=True):
        """
        Starts the new process.
        """
        self.logger.info('Start new Process in NOP Node [{}].'.format(self.name))
        if self.conversion_process is None:
            self.conversion_process = multiprocessing.Process(target=self.pass_data_to_next_node,
                                                              args=[self.reader])
            self.conversion_process.daemon = True
            self.conversion_process.start()
        super(SpawnNopNode, self).start_processing(recurse)

    def stop_processing(self, recurse=True):
        """
        Stops the new process.
        """
        super(SpawnNopNode, self).stop_processing(recurse)

        self.logger.info('Stopping process initiated by [{}].'.format(self.name))
        if self.conversion_process is not None:
            self.conversion_process.terminate()
        self.conversion_process = None
