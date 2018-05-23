# -*- coding: utf-8 -*-
# file: main.py
# author: joddiyzhang@gmail.com
# time: 2018/5/19 8:48 PM
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------
from src.preprocess.pp_mal_conv import PPMalConv
from src.test.t_auc import TAUC
from src.train.t_mal_conv import TMalConv
from src.train.t_mal_conv_ensemble import TMalConvEnsemble
from src.train.t_mal_conv_ensemble_feature import TMalConvEnsembleFeature
from src.train.t_mal_conv_feature import TMalConvFeature

if __name__ == '__main__':
    t_instance = TAUC()
    t_instance.run()
    # pp_instance = PPMalConv()
    # pp_instance.read_input()
