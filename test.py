# -*- coding: utf-8 -*-
# file: test.py
# author: joddiyzhang@gmail.com
# time: 2018/5/21 5:18 PM
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------

import numpy as np
import pandas as pd
# from keras.models import load_model
from keras.models import load_model
from sklearn.metrics import roc_auc_score, confusion_matrix

#
def get_bytes_array(data):
    """
    int to bytes array
    :param data:
    :return:
    """
    bytes_data = bytes(map(int, data.split(",")))
    bytes_data = crop_exceed_data(bytes_data)
    return [int(single_byte) for single_byte in bytes_data]


def crop_exceed_data(data):
    if len(data) <= 8192:
        return data
    return data[0: 8192]


tmp_v = pd.read_csv("./hdd1/malware_data/test.csv", header=None, sep="|", names=['row_data'],
                    error_bad_lines=False)
tmp_v = tmp_v["row_data"].apply(lambda x: get_bytes_array(x))
v_x = pd.DataFrame(tmp_v.tolist())
v_y = pd.read_csv("./hdd1/malware_data/test_label.csv", header=None, error_bad_lines=False)
del tmp_v
print('Shape of the v_x data: ', v_x.shape)
print('Shape of the v_y data: ', v_y.shape)

model = load_model('./cache/b51f071681e13de00a641d77f6bf0046.h5')
y_pred = model.predict(v_x)

y_true = v_y
fp_np_index = np.where(y_true == 0)

auc = roc_auc_score(y_true, y_pred)

# false positive less than 0.1%
# want to get the recall

fp_np = y_pred[fp_np_index].shape[0]
thre_index = int(np.ceil(fp_np - fp_np * 0.001))

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

auc = roc_auc_score(y_true, y_pred)

print('\n')
print('fp_rate:' + str(fp_rate))
print('recall_rate:' + str(recall_rate))
print('auc:' + str(auc))
