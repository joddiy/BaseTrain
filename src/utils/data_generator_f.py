# -*- coding: utf-8 -*-
# file: data_generator.py
# author: joddiyzhang@gmail.com
# time: 2018/5/22 10:17 PM
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------

import keras
import numpy as np


class DataGeneratorF(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, datasets, labels, batch_size=32, dim=1024, shuffle=True):
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

    def reverse_bytes(self, original):
        """
        make the bytes inverse
        :param original:
        :return:
        """
        return original[::-1]

    def convert_int(self, str_bytes):
        """
        convert bytes to int
        :param str_bytes:
        :return:
        """
        return int.from_bytes(str_bytes, byteorder='big', signed=False)

    def decode_rich_sign(self, rich_sign):
        """
        decode the rich sign, use the last 4 bytes to xor each 4 bytes from the start to end
        :param rich_sign:
        :return:
        """
        key = rich_sign[-4:]
        rich_sign_d = bytearray()
        for i in range(len(rich_sign)):
            rich_sign_d.append(rich_sign[i] ^ key[i % 4])
        return bytes(rich_sign_d)

    def get_fixed_head(self, data):
        """
        select some useful parts from the whole PE head
        :param data:
        :return:
        """
        bytes_data = bytes(map(int, data.split(",")))
        # # mz head
        # mz_head = bytes_data[0:64]
        # # dos sub
        # ms_dos_sub = bytes_data[64:128]
        # decode rich sign
        rich_sign_end = bytes_data[128:].find(b'\x52\x69\x63\x68') + 136
        rich_sign = self.decode_rich_sign(bytes_data[128:rich_sign_end])
        # pe head
        pe_head_start = bytes_data[128:].find(b'\x50\x45\x00\x00') + 128
        pe_head = bytes_data[pe_head_start:pe_head_start + 24]
        # there are two types of image optional head, PE 32 and PE 32+
        other_head = bytes_data[pe_head_start + 24:]
        if other_head[0:2] == b'\x0b\x01':
            image_optional_head_end = 96
        else:
            image_optional_head_end = 112
        image_optional_head = other_head[0:image_optional_head_end]
        # data directory
        data_directory = other_head[image_optional_head_end: image_optional_head_end + 128]
        # append all above parts
        # fixed_head = mz_head + ms_dos_sub + rich_sign + pe_head + image_optional_head + data_directory
        fixed_head = rich_sign + pe_head + image_optional_head + data_directory
        # for each sections, just get the non-zero value
        number_of_sections = self.convert_int(self.reverse_bytes(pe_head[6:8]))
        for offset in range(number_of_sections):
            offset_sections_start = image_optional_head_end + 128 + 40 * offset
            # fixed_head += other_head[offset_sections_start: offset_sections_start + 28] + \
            #               other_head[offset_sections_start + 36:offset_sections_start + 40]
            fixed_head += other_head[offset_sections_start + 36:offset_sections_start + 40]
        return [int(single_byte) for single_byte in fixed_head]

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, self.dim), dtype=int)
        y = np.zeros(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            bytes_data = self.get_fixed_head(self.datasets[ID])
            if len(bytes_data) > self.dim:
                X[i, :] = bytes_data[:self.dim]
            else:
                X[i, 0:len(bytes_data)] = bytes_data

            # Store class
            y[i] = self.labels[ID]

        return X, y
