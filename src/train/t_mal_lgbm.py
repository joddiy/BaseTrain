# -*- coding: utf-8 -*-
# file: t_mal_lgbm.py
# author: joddiyzhang@gmail.com
# time: 2018/5/28 11:02 AM
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------

import hashlib
import json
import time

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from src.config.config import *
from src.config.config import CACHE_DIR
from src.preprocess.pp_mal_conv import PPMalConv
from src.train.train import Train


class TMalLgbm(Train):
    """
    train of mal conv
    """

    def __init__(self):
        self.train_df, self.label_df = PPMalConv().read_t()
        # self.v_x, self.v_y = PPMalConv().read_v()
        self.max_len = 8192
        self.history = None
        self.model = None
        self.p_md5 = None
        self.summary = {
            'input': DATA_CONFIG["mal_conv"]['train'],
            'time': time.time(),
            's_test_size': 0.05,
            's_random_state': 5242,
            'params': {
                'application': 'binary',
            },
        }

    def generate_p(self):
        """
        generate parameter for the summary
        :return:
        """
        pass

    def run(self):
        """
        :return:
        """
        self.summary_model()
        self.train()

    def summary_model(self):
        """
        summary this model
        :return:
        """
        self.p_md5 = hashlib.md5(json.dumps(self.summary, sort_keys=True).encode('utf-8')).hexdigest()
        with open(CACHE_DIR + self.p_md5 + '.json', 'w') as file_pi:
            json.dump(self.summary, file_pi)

    def get_p(self, key):
        """
        get the parameter from the summary
        :param key:
        :return:
        """
        return self.summary[key]

    def get_model(self):
        """
        get a model
        :return:
        """
        pass

    def read_model(self, file_path):
        """
        read a model from file
        :return:
        """
        pass

    def train(self):
        x_train, x_test, y_train, y_test = train_test_split(self.train_df, self.label_df,
                                                            test_size=self.get_p("s_test_size"),
                                                            random_state=self.get_p("s_random_state"))
        del self.train_df, self.label_df

        lgbm_dataset = lgb.Dataset(x_train, y_train.values.ravel())
        valid_sets = lgb.Dataset(x_test, y_test.values.ravel())
        model = lgb.train(self.get_p("params"), lgbm_dataset, 100, valid_sets=valid_sets, early_stopping_rounds=10)
        y_pred = model.predict(x_test)
        file_path = "./models/" + self.p_md5
        for i in range(model.best_iteration):
            model.save_model(file_path + "_" + i + ".h5", num_iteration=model.best_iteration)
        print("full score : %.5f" % roc_auc_score(y_test.values.ravel(), y_pred))
