# -*- coding: utf-8 -*-
# file: utils.py
# author: joddiyzhang@gmail.com
# time: 2018/5/20 12:45 PM
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------
import pickle

from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score


def save(obj, name):
    try:
        filename = open(name + ".pickle", "wb")
        pickle.dump(obj, filename)
        filename.close()
        return True
    except:
        return False


class RocAucEvaluation(Callback):
    def __init__(self, interval=1):
        super(Callback, self).__init__()
        val_data = self.validation_data
        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch + 1, score))
