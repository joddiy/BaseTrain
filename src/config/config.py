# -*- coding: utf-8 -*-
# file: weibo_config.py
# author: joddiyzhang@gmail.com
# time: 2017/9/6 下午9:45
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------

import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__ + "/../../"))

DATA_DIR = './input/'  # '/hdd1/malware_data/'

CACHE_DIR = PROJECT_DIR + '/cache/'

DATA_CONFIG = {
    'mal_conv': {
        'using_cache': False,
        'train': [
            # DATA_DIR + '1_train.csv',
            # DATA_DIR + '2_train.csv',
            # DATA_DIR + '3_train.csv',
            DATA_DIR + '4_train.csv',
        ],
        'label': [
            # DATA_DIR + '1_train_label.csv',
            # DATA_DIR + '2_train_label.csv',
            # DATA_DIR + '3_train_label.csv',
            DATA_DIR + '4_train_label.csv',
        ],
        'v_train': [
            DATA_DIR + 'test.csv'
        ],
        'v_label': [
            DATA_DIR + 'test_label.csv'
        ],
        'test': [
            DATA_DIR + ''
        ],
        'cleaned_data': CACHE_DIR + 'mal_conv.h5',
    }
}
