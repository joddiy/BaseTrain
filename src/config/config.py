# -*- coding: utf-8 -*-
# file: weibo_config.py
# author: joddiyzhang@gmail.com
# time: 2017/9/6 下午9:45
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------

import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__ + "/../../"))

DATA_DIR = PROJECT_DIR + '/input/'

CACHE_DIR = PROJECT_DIR + '/cache/'

DATA_CONFIG = {
    'mal_conv': {
        'train': DATA_DIR + 'train.csv',
        'label': DATA_DIR + 'train_label.csv',
        'test': DATA_DIR + '',
        'cleaned_data': CACHE_DIR + 'mal_conv.h5',
    }
}
