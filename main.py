# -*- coding: utf-8 -*-
# file: main.py
# author: joddiyzhang@gmail.com
# time: 2018/5/19 8:48 PM
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------
from src.preprocess.pp_mal_conv import PPMalConv

if __name__ == '__main__':
    pp_instance = PPMalConv()
    train, label = pp_instance.run()
    print('Shape of the train data: ', train.shape)
    print('Shape of the label data: ', label.shape)
