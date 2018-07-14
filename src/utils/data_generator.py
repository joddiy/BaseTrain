# -*- coding: utf-8 -*-
# file: data_generator.py
# author: joddiyzhang@gmail.com
# time: 2018/5/22 10:17 PM
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------

import keras
import numpy as np


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, datasets, labels, batch_size=32, dim=8192, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.datasets = datasets
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def crop_exceed_data(self, data):
        if len(data) <= 8192:
            return data
        return data[0: 8192]

    def get_bytes_array(self, data):
        """
        int to bytes array
        :param data:
        :return:
        """
        bytes_data = bytes(map(int, data.split(",")))
        bytes_data = self.crop_exceed_data(bytes_data)
        return [int(single_byte) for single_byte in bytes_data]

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, self.dim), dtype=float)
        y = np.zeros(self.batch_size, dtype=float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            bytes_data = self.get_bytes_array(self.datasets[ID])
            X[i, 0:len(bytes_data)] = bytes_data
            # Store class
            y[i] = self.labels[ID]

        return X, y
