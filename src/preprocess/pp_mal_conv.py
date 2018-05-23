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
        self.train = []
        self.label = []
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
