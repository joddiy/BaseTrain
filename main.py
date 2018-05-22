# -*- coding: utf-8 -*-
# file: main.py
# author: joddiyzhang@gmail.com
# time: 2018/5/19 8:48 PM
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------
from src.preprocess.pp_mal_conv import PPMalConv
from src.train.t_mal_conv import TMalConv
from src.train.t_mal_conv_ensemble import TMalConvEnsemble

if __name__ == '__main__':
    t_instance = TMalConvEnsemble()
    t_instance.run()
    # pp_instance = PPMalConv()
    # pp_instance.read_input()
