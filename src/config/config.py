# -*- coding: utf-8 -*-
# file: weibo_config.py
# author: joddiyzhang@gmail.com
# time: 2017/9/6 下午9:45
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------

import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__ + "/../../"))

DATA_DIR = PROJECT_DIR + '/input/'

DATA_CONFIG = {
    'train': 'train.csv',
    'label': 'train_label.csv',

    'test' : '',
}
