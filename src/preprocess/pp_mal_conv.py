# -*- coding: utf-8 -*-
# file: pp_mal_conv.py
# author: joddiyzhang@gmail.com
# time: 2018/5/19 7:49 PM
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------
from src.config.config import *
import pandas as pd

from src.preprocess.pre_process import PreProcess
import os.path
import numpy as np


def get_bytes_array(data):
    """
    int to bytes array
    :param data:
    :return:
    """
    bytes_data = bytes(map(int, data.split(",")))
    bytes_data = crop_exceed_data(bytes_data)
    return [int(single_byte) for single_byte in bytes_data]


def crop_exceed_data(data):
    if len(data) <= 8192:
        return data
    return data[0: 8192]


def reverse_bytes(original):
    """
    make the bytes inverse
    :param original:
    :return:
    """
    return original[::-1]


def convert_int(str_bytes):
    """
    convert bytes to int
    :param str_bytes:
    :return:
    """
    return int.from_bytes(str_bytes, byteorder='big', signed=False)


def decode_rich_sign(rich_sign):
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


def get_fixed_head(data):
    """
    select some useful parts from the whole PE head
    :param data:
    :return:
    """
    bytes_data = bytes(map(int, data.split(",")))
    # mz head
    mz_head = bytes_data[0:64]
    # dos sub
    ms_dos_sub = bytes_data[64:128]
    # decode rich sign
    rich_sign_end = bytes_data[128:].find(b'\x52\x69\x63\x68') + 136
    rich_sign = decode_rich_sign(bytes_data[128:rich_sign_end])
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
    fixed_head = mz_head + ms_dos_sub + rich_sign + pe_head + image_optional_head + data_directory
    # for each sections, just get the non-zero value
    number_of_sections = convert_int(reverse_bytes(pe_head[6:8]))
    for offset in range(number_of_sections):
        offset_sections_start = image_optional_head_end + 128 + 40 * offset
        fixed_head += other_head[offset_sections_start: offset_sections_start + 28] + \
                      other_head[offset_sections_start + 36:offset_sections_start + 40]
    return [int(single_byte) for single_byte in fixed_head]


class PPMalConv(PreProcess):
    """
    pre-process class
    """

    def __init__(self):
        """
        init
        """
        self.config = DATA_CONFIG["mal_conv"]
        self.train = []
        self.label = []
        self.test = None
        self.v_y = None
        self.v_x = None
        self.max_len = 1024

    def run(self):
        """
        call all functions
        :return:
        """
        if not self.config['using_cache']:
            self.read_input()
            self.feature_engineering()
        elif not os.path.isfile(self.config["cleaned_data"]):
            self.read_input()
            self.feature_engineering()
            self.save_cleaned_data()
        else:
            store = pd.HDFStore(self.config["cleaned_data"])
            self.train = store['train']
            self.label = store['label']
        return self.train, self.label, self.v_x, self.v_y

    def read_input(self):
        """
        read input data
        :return:
        """

        # train data
        for file_name in self.config["train"]:
            self.train.append(pd.read_csv(file_name, header=None, sep="|", index_col=None))
        self.train = pd.concat(self.train, ignore_index=True)[0]

        # train label
        for file_name in self.config["label"]:
            self.label.append(pd.read_csv(file_name, header=None, index_col=None))
        self.label = pd.concat(self.label, ignore_index=True)[0]

        print('Length of the data: ', len(self.train))
        return self.train, self.label

    def read_t(self):
        tmp_v = pd.read_csv(self.config["train"][0], header=None, sep="|", names=['row_data'],
                            error_bad_lines=False)
        tmp_v = tmp_v["row_data"].apply(lambda x: get_bytes_array(x))
        self.train = pd.DataFrame(tmp_v.tolist(), dtype=int)
        self.label = pd.read_csv(self.config["label"][0], header=None, error_bad_lines=False)
        del tmp_v
        print('Shape of the train data: ', self.train.shape)
        print('Shape of the label data: ', self.label.shape)
        return self.train, self.label

    def read_t_f(self):
        tmp_v = pd.read_csv(self.config["train"][0], header=None, sep="|", names=['row_data'],
                            error_bad_lines=False)
        self.train = np.zeros((tmp_v.shape[0], self.max_len), dtype=int)
        self.label = pd.read_csv(self.config["label"][0], header=None, error_bad_lines=False)
        print('Shape of the train data: ', self.train.shape)
        print('Shape of the label data: ', self.label.shape)

        for i, item in enumerate(tmp_v["row_data"]):
            # Store sample
            bytes_data = get_fixed_head(item)
            if len(bytes_data) > self.max_len:
                self.train[i, :] = bytes_data[:self.max_len]
            else:
                self.train[i, 0:len(bytes_data)] = bytes_data
        return self.train, self.label

    def read_v(self):
        tmp_v = pd.read_csv(self.config["v_train"][0], header=None, sep="|", names=['row_data'],
                            error_bad_lines=False)
        tmp_v = tmp_v["row_data"].apply(lambda x: get_bytes_array(x))
        self.v_x = pd.DataFrame(tmp_v.tolist())
        self.v_y = pd.read_csv(self.config["v_label"][0], header=None, error_bad_lines=False)
        del tmp_v
        print('Shape of the v_x data: ', self.v_x.shape)
        print('Shape of the v_y data: ', self.v_y.shape)
        return self.v_x, self.v_y

    def read_v_f(self):
        tmp_v = pd.read_csv(self.config["v_train"][0], header=None, sep="|", names=['row_data'],
                            error_bad_lines=False)
        self.v_x = np.zeros((tmp_v.shape[0], self.max_len), dtype=int)
        self.v_y = pd.read_csv(self.config["v_label"][0], header=None, error_bad_lines=False)
        print('Shape of the v_x data: ', self.v_x.shape)
        print('Shape of the v_y data: ', self.v_y.shape)

        for i, item in enumerate(tmp_v["row_data"]):
            # Store sample
            bytes_data = get_fixed_head(item)
            if len(bytes_data) > self.max_len:
                self.v_x[i, :] = bytes_data[:self.max_len]
            else:
                self.v_x[i, 0:len(bytes_data)] = bytes_data

        return self.v_x, self.v_y

    def get_v(self):
        """
        read test data
        :return:
        """
        # train data
        for file_name in self.config["v_train"]:
            self.v_x.append(pd.read_csv(file_name, header=None, sep="|", index_col=None))
        self.v_x = pd.concat(self.v_x, ignore_index=True)[0]

        # train label
        for file_name in self.config["v_label"]:
            self.v_y.append(pd.read_csv(file_name, header=None, index_col=None))
        self.v_y = pd.concat(self.v_y, ignore_index=True)[0]

        print('Length of the data: ', len(self.v_x))
        return self.v_x, self.v_y

    def feature_engineering(self):
        """
        feature engineering
        :return:
        """
        pass

    def save_cleaned_data(self):
        """
        save cleaned data
        :return:
        """
        store = pd.HDFStore(self.config["cleaned_data"])
        store['train'] = self.train
        store['label'] = self.label
