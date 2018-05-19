# -*- coding: utf-8 -*-
# file: t_mal_conv.py
# author: joddiyzhang@gmail.com
# time: 2018/5/19 9:28 PM
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------
import pickle

import keras
from keras import Input
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Embedding, Conv1D, Multiply, GlobalMaxPooling1D
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.preprocess.pp_mal_conv import PPMalConv
from src.train.train import Train


def gate_cnn(gate_cnn_input, kernel_size):
    """
    construct a gated cnn by the specific kernel size
    :param gate_cnn_input:
    :param kernel_size:
    :return:
    """
    conv1_out = Conv1D(256, 256, strides=256)(gate_cnn_input)
    conv2_out = Conv1D(256, 256, strides=256, activation="sigmoid")(gate_cnn_input)
    merged = Multiply()([conv1_out, conv2_out])
    gate_cnn_output = GlobalMaxPooling1D()(merged)
    return gate_cnn_output


class TMalConv(Train):
    """
    train of mal conv
    """

    def __init__(self):
        self.train, self.label = PPMalConv().run()
        self.history = None
        self.model = None
        self.max_len = self.train.shape[1]

    def run(self, *args):
        """
        :param args:
        :return:
        """
        pass

    def summary_model(self):
        """
        summary this model
        :return:
        """
        pass

    def get_model(self, *args):
        """
        get a model
        :param max_len:
        :param kernel_sizes:
        :return:
        """
        net_input = Input(shape=(self.max_len,))

        embedding_out = Embedding(256, 8, input_length=self.max_len)(net_input)
        merged = gate_cnn(embedding_out, 256)
        # # add several ensemble gated cnn kernels
        # for ks in kernel_sizes:
        #     merged = concatenate([merged, gate_cnn(embedding_out, ks)])

        dense_out = Dense(128)(merged)
        # dropout_out = Dropout(0.2)(dense_out)
        net_output = Dense(1, activation='sigmoid')(dense_out)

        model = keras.models.Model(inputs=net_input, outputs=net_output)
        model.summary()

        return model

    def train(self):
        batch_size = 256
        epochs = 16

        self.model = self.get_model(self.max_len)

        x_train, x_test, y_train, y_test = train_test_split(self.train, self.label,
                                                            test_size=0.01, random_state=5242)

        callback = EarlyStopping("val_loss", patience=3, verbose=0, mode='auto')

        self.model.compile(loss='binary_crossentropy',
                           optimizer='sgd',
                           metrics=['accuracy'])

        self.history = self.model.fit(x_train, y_train,
                                      batch_size=batch_size,
                                      epochs=epochs, callbacks=[callback],
                                      validation_data=(x_test, y_test))

        score = roc_auc_score(y_test, self.model.predict(x_test))
        print('Auc score:', score)

    def save_history(self):
        """

        :return:
        """
        with open('/trainHistoryDict', 'wb') as file_pi:
            pickle.dump(self.history, file_pi)

    def save_model(self):
        """

        :return:
        """
        self.model.save('model.h5')
