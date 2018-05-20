# -*- coding: utf-8 -*-
# file: t_mal_conv.py
# author: joddiyzhang@gmail.com
# time: 2018/5/19 9:28 PM
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------
import hashlib
import json
import pickle

import keras
from keras import Input
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Embedding, Conv1D, Multiply, GlobalMaxPooling1D
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.config.config import CACHE_DIR
from src.preprocess.pp_mal_conv import PPMalConv
from src.train.train import Train
from src.utils.utils import save


class TMalConv(Train):
    """
    train of mal conv
    """

    def __init__(self):
        self.train_df, self.label_df = PPMalConv().run()
        self.max_len = self.train_df.shape[1]
        self.history = None
        self.model = None
        self.p_md5 = None
        self.summary = {
            'batch_size': 256,
            'epochs': 64,
            's_test_size': 0.01,
            's_random_state': 5242,
            'e_s_patience': 8,
            'g_c_filter': 256,
            'g_c_kernel_size': 256,
            'g_c_stride': 256,
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

        x_train, x_test, y_train, y_test = train_test_split(self.train_df, self.label_df,
                                                            test_size=self.get_p("s_test_size"),
                                                            random_state=self.get_p("s_random_state"))

        callback = EarlyStopping("val_loss", patience=self.get_p("e_s_patience"), verbose=0, mode='auto')

        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        h = self.model.fit(x_train, y_train,
                                      batch_size=batch_size,
                                      epochs=epochs, callbacks=[callback],
                                      validation_data=(x_test, y_test))
        self.history = h.history

        score = roc_auc_score(y_test, self.model.predict(x_test))
        print('Auc score:', score)

    def save_history(self):
        """

        :return:
        """
        save(self.history, CACHE_DIR + self.p_md5)

    def save_model(self):
        """

        :return:
        """
        self.model.save(CACHE_DIR + self.p_md5 + '.h5')
