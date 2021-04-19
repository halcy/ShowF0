import numpy as np
import h5py
from . import Node,Receiver
import time


class HDF5_Receiver(Node.Node):
    def __init__(self, path_name, dataset_name, chunksize ,name='HDf5_Receiver'):
        """
        Initializes node used to receive data and write it to .h5 file in real time.

        Parameters:
            path_name(str) - The filename of the HDF5 file that is written.
            dataset_name - The name of the dataset inside the HDF5-file
            chunksize - The chunksize of data that is written to disk at once. This limits disk writing
                        and can be calculated as <samplerate>*<writing interval in seconds>.
        """

        super().__init__(name=name, has_inputs=True)
        self.path_name = path_name
        self.dataset_name = dataset_name
        self.data_shape = None
        self.timer = Receiver.Receiver()

        self.chunk_size = chunksize
        #print(self.chunk_size)

    def add_data(self, data_frame, data_id=0):
        #print(data_frame.shape)
        start = time.time()
        frame_shape = np.shape(data_frame)
        # 2D-ify if needed
        if len(frame_shape) == 1:
            data_frame = np.array(data_frame).reshape(-1, 1)
            frame_shape = np.shape(data_frame)

        # create file on first call
        if self.data_shape is None:
            self.data_shape = frame_shape
            chunksize = (self.data_shape[0] * 100,) + self.data_shape[1:]
            with h5py.File(self.path_name, 'w') as hdf_handle:
                dset = hdf_handle.create_dataset(self.dataset_name, shape=self.data_shape,
                                                 maxshape=(None,) + self.data_shape[1:],
                                                 chunks=chunksize)
                # write data
                dset[0:frame_shape[0]] = data_frame
            end = time.time()

        else:
            with h5py.File(self.path_name, 'a') as f:
                dset = f[self.dataset_name]
                # resize dataset
                dset.resize((dset.shape[0] + frame_shape[0],) + self.data_shape[1:])
                # write data
                dset[-frame_shape[0]:] = data_frame
            end = time.time()
        self.timer.add_data(end - start)
