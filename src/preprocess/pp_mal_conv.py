# -*- coding: utf-8 -*-
# file: pp_mal_conv.py
# author: joddiyzhang@gmail.com
# time: 2018/5/19 7:49 PM
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------
from src.config.config import *
import pandas as pd

from src.preprocess.pre_process import PreProcess


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

    def run(self):
        """
        call all functions
        :return:
        """
        try:
            store = pd.HDFStore(self.config["cleaned_data"])
            self.train = store['train']
            self.label = store['label']
        except Exception as e:
            self.read_input()
            self.feature_engineering()
            self.save_cleaned_data()
        return self.train, self.label

    def read_input(self):
        """
        read input data
        :return:
        """

        self.train = pd.read_csv(self.config["train"], header=None, sep="|", names=['row_data'], error_bad_lines=False)
        self.label = pd.read_csv(self.config["label"], sep=",", error_bad_lines=False)

    def feature_engineering(self):
        """
        feature engineering
        :return:
        """
        self.label.drop('sample_id', axis=1, inplace=True)
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
