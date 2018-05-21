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


class PPMalConv(PreProcess):
    """
    pre-process class
    """

    def __init__(self):
        """
        init
        """
        self.config = DATA_CONFIG["mal_conv"]
        self.train = None
        self.label = None
        self.test = None
        self.v_y = None
        self.v_x = None

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

    def get_v(self):
        """

        :return:
        """
        self.v_x = pd.read_csv(self.config["v_train"], header=None, names=range(8192), error_bad_lines=False)
        self.v_x.fillna(0)
        self.v_x.astype("float64")
        self.v_y = pd.read_csv(self.config["v_label"], header=None, error_bad_lines=False)
        print('Shape of the v_x data: ', self.v_x.shape)
        print('Shape of the v_y data: ', self.v_y.shape)
        return self.v_x, self.v_y

    def read_input(self):
        """
        read input data
        :return:
        """
        for idx in range(len(self.config["train"])):
            train_file = self.config["train"][idx]
            label_file = self.config["label"][idx]
            tmp_train = pd.read_csv(train_file, header=None, names=range(8192), error_bad_lines=False)
            tmp_train.fillna(0)
            tmp_label = pd.read_csv(label_file, header=None, error_bad_lines=False)
            if self.train is None:
                self.train = tmp_train
            else:
                self.train = self.train.append(tmp_train)
            if self.label is None:
                self.label = tmp_label
            else:
                self.label = self.label.append(tmp_label)
            print('Shape of the train data: ', self.train.shape)
            print('Shape of the label data: ', self.label.shape)

        self.v_x = pd.read_csv(self.config["v_train"], header=None, names=range(8192), error_bad_lines=False)
        self.v_x.fillna(0)
        self.v_y = pd.read_csv(self.config["v_label"], header=None, error_bad_lines=False)
        print('Shape of the v_x data: ', self.v_x.shape)
        print('Shape of the v_y data: ', self.v_y.shape)

    def read_v(self):
        self.v_x = pd.read_csv(self.config["v_train"], header=None, names=range(8192), error_bad_lines=False)
        self.v_x.fillna(0)
        self.v_y = pd.read_csv(self.config["v_label"], header=None, error_bad_lines=False)
        print('Shape of the v_x data: ', self.v_x.shape)
        print('Shape of the v_y data: ', self.v_y.shape)
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
