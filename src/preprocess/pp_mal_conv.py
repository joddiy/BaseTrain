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
        return self.train, self.label

    def read_input(self):
        """
        read input data
        :return:
        """
        self.train = pd.read_csv(self.config["train"], header=None, sep="|", names=['row_data'], error_bad_lines=False)
        for idx in range(len(self.config["train"])):
            train_file = self.config["train"][idx]
            label_file = self.config["label"][idx]
            print(train_file)
            print(label_file)
        #
        # self.label = pd.read_csv(self.config["label"], sep=",", header=None, names=['label'], error_bad_lines=False)
        # self.v_x = pd.read_csv(self.config["v_x"], header=None, sep="|", names=['row_data'], error_bad_lines=False)
        # self.v_y = pd.read_csv(self.config["v_y"], sep=",", error_bad_lines=False)

    def feature_engineering(self):
        """
        feature engineering
        :return:
        """
        self.train = self.train["row_data"].apply(lambda x: get_bytes_array(x))
        self.train = pd.DataFrame(self.train.tolist())
        self.train = self.train["row_data"].apply(lambda x: get_bytes_array(x))
        self.train = pd.DataFrame(self.train.tolist())

    def save_cleaned_data(self):
        """
        save cleaned data
        :return:
        """
        store = pd.HDFStore(self.config["cleaned_data"])
        store['train'] = self.train
        store['label'] = self.label
