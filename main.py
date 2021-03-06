# -*- coding: utf-8 -*-
# file: main.py
# author: joddiyzhang@gmail.com
# time: 2018/5/19 8:48 PM
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------
from src.preprocess.pp_mal_conv import PPMalConv
from src.train.t_ember_lgbm import TMalLgbmEmber
from src.train.t_mal_conv import TMalConv
from src.train.t_mal_conv_ensemble import TMalConvEnsemble
from src.train.t_mal_conv_ensemble_feature import TMalConvEnsembleFeature
from src.train.t_mal_conv_feature import TMalConvFeature
from src.train.t_mal_lgbm import TMalLgbm
from src.train.t_mal_lgbm_feature import TMalLgbmF


from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

from src.train.t_res_net import TResNet

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    t_instance = TMalConvEnsembleFeature()
    t_instance.run()
    # pp_instance = PPMalConv()
    # pp_instance.read_input()
