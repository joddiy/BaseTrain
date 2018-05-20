# -*- coding: utf-8 -*-
# file: t_mal_conv.py
# author: joddiyzhang@gmail.com
# time: 2018/5/21 12:25 AM
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------

import hashlib
import json

import keras
from keras import Input
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Embedding, Conv1D, Multiply, GlobalMaxPooling1D
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src.config.config import CACHE_DIR
from src.preprocess.pp_mal_conv import PPMalConv
from src.test.test import Test
from src.utils.utils import save
import numpy as np


class TMalConv(Test):
    def __init__(self):
        """

        """
        self.v_x = None
        self.v_y = None

    def predict(self):
        pass

    def run(self):
        self.v_x, self.v_y = PPMalConv().get_v()


