# -*- coding: utf-8 -*-
# file: t_auc.py
# author: joddiyzhang@gmail.com
# time: 2018/5/21 12:25 AM
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------

from os import listdir
from os.path import isfile, join

import numpy as np
from keras.models import load_model
from sklearn.metrics import roc_auc_score, confusion_matrix

from src.preprocess.pp_mal_conv import PPMalConv
from src.test.test import Test
from src.utils.utils import save


class TAUCF(Test):
    def __init__(self):
        """
        init
        """
        self.v_x, self.v_y = PPMalConv().read_v()

    def predict(self):
        pass

    def run(self):
        """

        :return:
        """
        y_true = self.v_y
        fp_np_index = np.where(y_true == 0)
        model_dir = './backup/'

        model_files = [f for f in listdir(model_dir) if isfile(join(model_dir, f)) and f[-3:] == '.h5']

        for f_name in model_files:

            model = load_model(model_dir + f_name)
            y_pred = model.predict(self.v_x)

            auc = roc_auc_score(y_true, y_pred)
            print('\n')
            print(f_name)
            print('auc:' + str(auc))

            res = {}

            for idx in range(100, 1000):
                fp_np = y_pred[fp_np_index].shape[0]
                thre_index = int(np.ceil(fp_np - fp_np * idx / 10000))

                sorted_pred_prob = np.sort(y_pred[fp_np_index], axis=0)
                thre = sorted_pred_prob[thre_index]
                if thre == 1:
                    thre = max(sorted_pred_prob[np.where(sorted_pred_prob != 1)])

                y_pred_prob = np.vstack((y_pred.transpose(), (1 - y_pred).transpose())).transpose()
                y_pred_prob[:, 1] = thre
                y_pred_label = np.argmin(y_pred_prob, axis=-1)

                tn, fp, fn, tp = confusion_matrix(y_true, y_pred_label).ravel()
                fp_rate = fp / (fp + tn)
                recall_rate = tp / (tp + fn)

                res[fp_rate] = recall_rate
                if idx % 100 == 0:
                    print('thre:', thre)
                    print("fp: ", fp_rate)
                    print("recall: ", recall_rate)

            save(res, './src/auc/' + f_name[0:-3])
            # lists = sorted(object_file.items())
            # x, y = zip(*lists)
            # plt.title('model accuracy')
            # plt.ylabel('recall')
            # plt.xlabel('fp rate')
            # plt.show()
