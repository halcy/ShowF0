import logging
import os
import sys
import errno
import random
import os.path

import time
import numpy as np
import multiprocessing
import gc

from . import Node

import watchdog
import watchdog.events
import watchdog.observers

class ChangeWatcher(watchdog.events.FileSystemEventHandler):
    def __init__(self, watch_path):
        self.changed = multiprocessing.Value('b', False)
        self.watch_path = os.path.normpath(watch_path)
        self.watch_dir = os.path.dirname(self.watch_path)
        print(self.watch_dir)
        
        self.observer = watchdog.observers.Observer()
        self.observer.schedule(self, path=self.watch_dir, recursive=False)
        self.observer.start()
        
    def check_change(self):
        if self.changed.value == True:
            self.changed.value = False
            return True
        else:
            return False
        
    def on_modified(self, event):
        changed_path = os.path.normpath(event.src_path)
        if self.watch_path == changed_path:
            self.changed.value = True
    
    def stop(self):
        if not self.observer is None:
            self.observer.stop()
            del self.observer
            self.observer = None
    
    def __del__(self):
        self.stop()
        
class DNN2(Node.Node):
    """
    Basic Keras neural network, functions for training and running
    
    Does not accumulate training data - do that externally
    """
    def __init__(self, model_source = None, train_log_dir = None, name = "DNN2", auto_reload = False, only_final=False, denorm_out=True):
        """
        Initialize a DNN.
        
        model_source - file name of a model file to load, or a
                       function that will return a keras model to
                       be trained
        train_log_dir - directory for training logs and intermediate
                        files
        """ 
        super().__init__(name = name)
        self.model_source = model_source
        self.train_log_dir = train_log_dir
        
        self.feat_in_pipe_out, self.feat_in_pipe_in = multiprocessing.Pipe(False)
        self.mapping_process = None
        self.run_map = multiprocessing.Value('b', False)
        self.auto_reload = auto_reload
        self.observer = None
        self.only_final = only_final
        self.denorm_out = denorm_out
        
    def train_network(self, data_x, data_y, optimizer_generator, num_epochs, batch_size, out_name, validation_split = 0.05, shuffle_data = True, loss="mse", metrics=None, center_out=True):
        """
        Trains for some epochs with some data
        
        data_x, data_y: numpy array with training data, timesteps x dimensions
        optimizer_generator: function that returns an optimizer
        num_epochs: how many epochs to train for
        batch_size: minibatch size for training
        out_name: file name of where the trained network should go
        validation_split: what fraction of data should be used for validation (default: 0.05)?
        shuffle_data: should the data be shuffled before use (default: True)?
        
        Returns training history
        """ 
        training_process = multiprocessing.Process(target = self.train_network_process, args = (data_x, data_y, optimizer_generator, num_epochs, batch_size, out_name, validation_split, shuffle_data, loss, metrics, center_out))
        training_process.start()
        history = training_process.join()
        
        # Set model source to out_name so that next operation will run on the new model
        self.model_source = out_name
        
        return history

    def train_network_process(self, data_x, data_y, optimizer_generator, num_epochs, batch_size, out_name, validation_split, shuffle_data, loss, metrics, center_out):
        """
        Trains the neural network.
        
        Parameters as in train_network
        """
        import keras
        sys.stdout.flush()
        keras.backend.clear_session()
        sys.stdout.flush()

        # Load data
        train_data_in = np.load(data_x)
        train_data_out = np.load(data_y)
        
        # Shuffle if desired
        if shuffle_data:
            random.seed(235) # TODO data dependent better?
            samples_labels = list(zip(train_data_in, train_data_out))
            random.shuffle(samples_labels)
            train_data_in, train_data_out = zip(*samples_labels)

        # Split into validation and training
        samples_val = int(len(train_data_in) * validation_split) 
        
        val_x = np.array(train_data_in[:samples_val])
        val_y = np.array(train_data_out[:samples_val])
        
        train_x = np.array(train_data_in[samples_val:])
        train_y = np.array(train_data_out[samples_val:])
        
        # Make sure training log directory exists
        os.makedirs(self.train_log_dir, exist_ok = True)
        
        # Set up model checkpointing and training logging
        cp_all = keras.callbacks.ModelCheckpoint(
            os.path.join(self.train_log_dir, 'weights.latest.hdf5'), 
            monitor='val_loss', 
            verbose=0, 
            save_best_only=False, 
            save_weights_only=False, 
            mode='auto', 
            period=1
        )

        cp_best = keras.callbacks.ModelCheckpoint(
            os.path.join(self.train_log_dir, 'model_best.h5'),
            monitor='val_loss', 
            verbose=0, 
            save_best_only=True, 
            save_weights_only=False, 
            mode='auto', 
            period=1
        )
        
        cp_tensor_board = keras.callbacks.TensorBoard(
            log_dir = os.path.join(self.train_log_dir, 'tb_logs/base'),
            histogram_freq = 0, 
            write_graph = True, 
            write_images = True, 
            embeddings_freq = 0, 
            embeddings_layer_names = None,
            embeddings_metadata = None
        )
        
        # Perform normalization setup and normalize
        logging.info("Determining data set parameters...")
        x_mean_std = [np.mean(train_x, axis = 0), np.std(train_x, axis = 0)]
        y_mean_std = [np.mean(train_y, axis = 0), np.std(train_y, axis = 0)]
                
        logging.info("Normalizing...")
        train_x = np.nan_to_num((train_x - x_mean_std[0]) / x_mean_std[1])
        val_x = np.nan_to_num((val_x - x_mean_std[0]) / x_mean_std[1])        
        if center_out == True:
            train_y = np.nan_to_num((train_y - y_mean_std[0]) / y_mean_std[1])
            val_y = np.nan_to_num((val_y - y_mean_std[0]) / y_mean_std[1])
        #print("y:", train_y[:500,:])
        
        gc.collect()
        
        # Create / load model and optimizer
        optimizer = optimizer_generator()
        model = None
        if isinstance(self.model_source, str):
            model = keras.models.load_model(self.model_source)
            if self.only_final == True:
                print("Setting layer trainability for adapt")
                for layer in model.layers:
                    layer.trainable = False
                model.layers[-1].trainable = True
                for layer in model.layers:
                    print(layer.trainable)
            print("Loaded model, not recompiling")
        else:
            model = self.model_source(train_x.shape[-1], train_y.shape[-1])
            # Set up network and trainer for training
            model.compile(
                loss=loss,
                optimizer=optimizer,
                metrics = metrics
            )
        
        # Train
        checkpoints = [cp_all, cp_best, cp_tensor_board]
        
        logging.info("Neural network training begins")
        history = model.fit(
            train_x,
            train_y,
            batch_size = batch_size,
            validation_data = (val_x, val_y),
            epochs = num_epochs,
            verbose = 1,
            callbacks = checkpoints,
        )
        
        # Store model and normalization data TODO: potentially, when retraining, only adapt these
        model.save(out_name)
        np.save(out_name + "_x_mean.npy", x_mean_std[0])
        np.save(out_name + "_x_std.npy", x_mean_std[1])
        np.save(out_name + "_y_mean.npy", y_mean_std[0])
        np.save(out_name + "_y_std.npy", y_mean_std[1])
        
        sys.stdout.flush()
        keras.backend.clear_session()
        sys.stdout.flush()
        
        return history
      
      
    def train_maml(self, data_x, data_y, num_epochs, batch_size, out_name, lr_inner=0.01, log_steps=10):
    
        """
        Trains the neural network.
        
        Parameters as in train_network
        """
        import tensorflow.keras as keras
        import tensorflow as tf
        import tensorflow.keras.backend as keras_backend
        import tempfile
        
        sys.stdout.flush()
        keras.backend.clear_session()
        sys.stdout.flush()

        # Load data
        train_x = np.load(data_x)
        train_y = np.load(data_y)
        
        # Make sure training log directory exists
        os.makedirs(self.train_log_dir, exist_ok = True)
        
        # Perform normalization setup and normalize
        logging.info("Determining data set parameters...")
        x_mean_std = [np.mean(train_x, axis = 0), np.std(train_x, axis = 0)]
        y_mean_std = [np.mean(train_y, axis = 0), np.std(train_y, axis = 0)]
                
        logging.info("Normalizing...")
        train_x = np.nan_to_num((train_x - x_mean_std[0]) / x_mean_std[1])
        train_y = np.nan_to_num((train_y - y_mean_std[0]) / y_mean_std[1])
        gc.collect()
        
        # Create / load model and optimizer
        optimizer = keras.optimizers.Adam(clipnorm=0.999)
        model = None
        if isinstance(self.model_source, str):
            model = keras.models.load_model(self.model_source)
        else:
            model = self.model_source(train_x.shape[-1], train_y.shape[-1])
            model.compile(optimizer=optimizer, loss="mse")
        
        logging.info("Neural network training begins (MAML)")
        losses = []
        for ep in range(num_epochs):
            #print(train_x.shape)
            #print(train_y.shape)
            
            # Flush tensorflow graph to free tf memory
            sys.stdout.flush()
            keras.backend.clear_session()
            sys.stdout.flush()
            gc.collect()

            # Important: No shuffling (tasks are in memory in order)

            total_loss = 0
            start = time.time()
            
            # Go through all "tasks" (the entire training set)
            for i in range(0, len(train_x), batch_size):
                # Sample tasks (Line 3, Line 5) - effectively, just select a batch
                # Batch size must be <= task size for this to work
                x = tf.convert_to_tensor(train_x[i:i+batch_size,:])
                y = tf.convert_to_tensor(train_y[i:i+batch_size,:])
                #print(x.shape, y.shape)
                
                # Begin calculation of "outer" loss
                #model(x, training=True) # Forward pass is needed because tensorflow breaks if omitted
                with tf.GradientTape() as test_tape:
                    # Begin calculation of "inner" loss ()
                    with tf.GradientTape() as train_tape:
                        # Calculate the inner loss (loss with regard to current model, line 6)
                        logits = model(x, training=True)
                        train_loss = keras.backend.mean(keras.losses.mean_squared_error(y, logits))
                        
                    # Find gradient so we can perform inner weight update
                    gradients = train_tape.gradient(train_loss, model.trainable_variables)
                    
                    # Now, make a copy of the model to work with, weights copied from current model
                    model_copy = None
                    if isinstance(self.model_source, str):
                        model_copy = keras.models.load_model(self.model_source)
                    else:
                        model_copy = self.model_source(train_x.shape[-1], train_y.shape[-1])
                
                    model_copy(x)
                    model_copy.set_weights(model.get_weights())
                    
                    # Update this model with the gradients from before. Skip parameterless layers (dropout!)
                    k = 0            
                    for j in range(len(model_copy.layers)):
                        if hasattr(model_copy.layers[j], 'kernel'):
                            model_copy.layers[j].kernel = tf.subtract(model.layers[j].kernel, tf.multiply(lr_inner, gradients[k]))
                            model_copy.layers[j].bias = tf.subtract(model.layers[j].bias, tf.multiply(lr_inner, gradients[k+1]))
                            k += 2
                        else:
                            pass

                    # Now, calculate the outer loss, with regard to the updated model (line 8)
                    logits = model_copy(x, training=True)
                    test_loss = keras.backend.mean(keras.losses.mean_squared_error(y, logits))

                # Use these with the optimizer to update the current model (also line 8)
                gradients = test_tape.gradient(test_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                # Logging
                total_loss += test_loss
                loss = total_loss / (i/batch_size+1.0)
                losses.append(loss)

                if i % (log_steps * batch_size) == 0:
                    print('Step {} / {}: loss = {}, Time to run {} steps = {}'.format(ep, i, loss, log_steps, time.time() - start))
                    logging.info('Step {} / {}: loss = {}, Time to run {} steps = {}'.format(ep, i, loss, log_steps, time.time() - start))
                    start = time.time()
                    gc.collect()
            
        # Store model and normalization data TODO: potentially, when retraining, only adapt these
        model.save(out_name)
        np.save(out_name + "_x_mean.npy", x_mean_std[0])
        np.save(out_name + "_x_std.npy", x_mean_std[1])
        np.save(out_name + "_y_mean.npy", y_mean_std[0])
        np.save(out_name + "_y_std.npy", y_mean_std[1])
        
        sys.stdout.flush()
        keras.backend.clear_session()
        sys.stdout.flush()
        
        return losses
    
    def add_data(self, sample, data_id):
        """
        Add data for decoding. Expected to have two dimensions.
        """
        sample = np.array(sample)
        if len(sample.shape) == 1:
            sample = sample.reshape(1, -1)
        self.feat_in_pipe_in.send(sample)   
        
    def mapping_runner(self): 
        """
        Decodes frames and calls callbacks.
        
        At this point, model_source MUST be a loadable model
        """
        if not isinstance(self.model_source, str):
            raise ValueError("Tried to initialize mapping without a model")
        
        import keras
        sys.stdout.flush()
        keras.backend.clear_session()
        sys.stdout.flush()
        
        # Load model and normalization parameters
        print("Loading model", self.model_source)
        model = keras.models.load_model(self.model_source)
        x_mean_std = [np.load(self.model_source + "_x_mean.npy"), np.load(self.model_source + "_x_std.npy")]
        y_mean_std = [np.load(self.model_source + "_y_mean.npy"), np.load(self.model_source + "_y_std.npy")]
        
        while self.run_map.value:
            sample = self.feat_in_pipe_out.recv()
            
            # Reload if needed
            if self.auto_reload == True and self.observer.check_change():
                model = keras.models.load_model(self.model_source)
            
            # Normalize
            sample = (sample - x_mean_std[0]) / x_mean_std[1]
            
            # Map
            result_sample = model.predict(sample, batch_size = 1)

            # Denormalize
            if self.denorm_out:
                result_sample = (result_sample * y_mean_std[1]) + y_mean_std[0]
            
            self.output_data(result_sample)
        
        sys.stdout.flush()
        keras.backend.clear_session()
        sys.stdout.flush()
        
    def start_processing(self,recurse = True):
        """
        Starts the decode process. train_mode has to be False to work.
        """
        if isinstance(self.model_source, str) and self.auto_reload:
            self.observer = ChangeWatcher(self.model_source)
            
        if self.mapping_process is None:
            self.run_map.value = True
            self.mapping_process = multiprocessing.Process(target = self.mapping_runner)
            self.mapping_process.start()
        super(DNN2, self).start_processing(recurse)
        
    def stop_processing(self, recurse = True):
        """
        Stops the streaming process.
        """
        super(DNN2, self).stop_processing(recurse)
        self.run_map.value = False
        if not self.mapping_process is None: # TODO sleep before murdering?
            self.mapping_process.terminate()
        self.mapping_process = None
        self.observer.stop()
        del self.observer
