# -*- coding: utf-8 -*-
# file: t_mal_conv.py
# author: joddiyzhang@gmail.com
# time: 2018/5/19 9:28 PM
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------
import hashlib
import json
import time

import keras
import pandas as pd
from keras import Input
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Dense, Embedding, Conv1D, Multiply, GlobalMaxPooling1D
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from src.config.config import *

from src.config.config import CACHE_DIR
from src.preprocess.pp_mal_conv import PPMalConv
from src.train.train import Train
from src.utils.data_generator import DataGenerator
from src.utils.utils import save
import numpy as np


class TMalConv(Train):
    """
    train of mal conv
    """

    def __init__(self):
        self.train_df, self.label_df = PPMalConv().read_input()
        self.v_x = None
        self.v_y = None
        self.max_len = 8192
        self.history = None
        self.model = None
        self.p_md5 = None
        self.summary = {
            'input': DATA_CONFIG["mal_conv"]['train'],
            'batch_size': 32,
            'epochs': 6,
            's_test_size': 0.05,
            's_random_state': 5242,
            'e_s_patience': 3,
            'g_c_filter': 128,
            'g_c_kernel_size': 500,
            'g_c_stride': 500,
            # 'fp_rate': [0.001, 0.005, 0.01, 0.05],
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
        self.train()
        self.summary_model()
        self.save_model()
        self.save_history()

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

    def gate_cnn(self, gate_cnn_input):
        """
        construct a gated cnn by the specific kernel size
        :param gate_cnn_input:
        :param kernel_size:
        :return:
        """
        conv1_out = Conv1D(self.get_p("g_c_filter"), self.get_p("g_c_kernel_size"), strides=self.get_p("g_c_stride"))(
            gate_cnn_input)
        conv2_out = Conv1D(self.get_p("g_c_filter"), self.get_p("g_c_kernel_size"), strides=self.get_p("g_c_stride"),
                           activation="sigmoid")(gate_cnn_input)
        merged = Multiply()([conv1_out, conv2_out])
        gate_cnn_output = GlobalMaxPooling1D()(merged)
        return gate_cnn_output

    def get_model(self):
        """
        get a model
        :param max_len:
        :param kernel_sizes:
        :return:
        """
        net_input = Input(shape=(self.max_len,))

        embedding_out = Embedding(256, 8, input_length=self.max_len)(net_input)
        merged = self.gate_cnn(embedding_out)

        dense_out = Dense(128)(merged)
        net_output = Dense(1, activation='sigmoid')(dense_out)

        model = keras.models.Model(inputs=net_input, outputs=net_output)
        model.summary()

        return model

    def train(self):
        batch_size = self.get_p("batch_size")
        epochs = self.get_p("epochs")

        self.model = self.get_model()

        partition_train, partition_validation = train_test_split(range(len(self.train_df)), test_size=0.05)
        print('Length of the train: ', len(partition_train))
        print('Length of the validation: ', len(partition_validation))

        callback = TensorBoard(log_dir='./logs/{}'.format(time.time()), batch_size=batch_size)
        # callback = EarlyStopping("val_loss", patience=self.get_p("e_s_patience"), verbose=0, mode='auto')

        # Generators
        training_generator = DataGenerator(partition_train, self.train_df, self.train_df, batch_size)
        validation_generator = DataGenerator(partition_validation, self.train_df, self.train_df, batch_size)

        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        h = self.model.fit_generator(generator=training_generator,
                                     validation_data=validation_generator,
                                     use_multiprocessing=True,
                                     epochs=epochs,
                                     workers=6,
                                     callbacks=[callback])
        self.history = h.history

    def save_history(self):
        """

        :return:
        """
        # self.get_fp()
        with open(CACHE_DIR + self.p_md5 + '.json', 'w') as file_pi:
            json.dump(self.summary, file_pi)
        save(self.history, CACHE_DIR + self.p_md5)

    def save_model(self):
        """

        :return:
        """
        self.model.save(CACHE_DIR + self.p_md5 + '.h5')

    def get_fp(self):
        """

        :return:
        """
        self.v_x, self.v_y = PPMalConv().read_v()
        y_true = self.v_y
        fp_np_index = np.where(y_true == 0)

        y_pred = self.model.predict(self.v_x)
        auc = roc_auc_score(y_true, y_pred)

        sub = pd.DataFrame()
        sub['sample_id'] = range(len(y_pred))
        sub['malware'] = y_pred
        sub.to_csv(CACHE_DIR + self.p_md5 + '.csv', header=None, index=False)

        fp_np = y_pred[fp_np_index].shape[0]
        for idx in range(len(self.get_p("fp_rate"))):
            print('\n, fp: ', self.get_p("fp_rate")[idx])
            thre_index = int(np.ceil(fp_np - fp_np * self.get_p("fp_rate")[idx]))

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

            self.history['fp_rate'] = str(fp_rate)
            self.history['recall_rate'] = str(recall_rate)
            self.history['auc'] = str(auc)

            print('\n')
            print('fp_rate:' + str(fp_rate))
            print('recall_rate:' + str(recall_rate))
            print('auc:' + str(auc))
