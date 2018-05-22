# -*- coding: utf-8 -*-
# file: t_mal_conv.py
# author: joddiyzhang@gmail.com
# time: 2018/5/19 9:28 PM
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------
import hashlib
import json
import time

from src.config.config import *

import keras
from keras import Input
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Dense, Embedding, Conv1D, Multiply, GlobalMaxPooling1D, concatenate, Dropout
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

from src.config.config import CACHE_DIR
from src.preprocess.pp_mal_conv import PPMalConv
from src.train.train import Train
from src.utils.data_generator import DataGenerator
from src.utils.utils import save
import numpy as np


class TMalConvEnsemble(Train):
    """
    train of mal conv
    """

    def __init__(self):
        self.train_df, self.label_df = PPMalConv().read_input()
        self.v_x = None
        self.v_y = None
        self.max_len = self.train_df.shape[1]
        self.history = None
        self.model = None
        self.p_md5 = None
        self.summary = {
            'input': DATA_CONFIG["mal_conv"]['train'],
            'batch_size': 32,
            'epochs': 9,
            's_test_size': 0.05,
            's_random_state': 1234,
            'e_s_patience': 2,
            'gate_units': [
                [32, 1, 1],
                [32, 2, 1],
                [32, 4, 1],
                [32, 8, 1],
                [32, 16, 1],
                [32, 32, 1],
                [32, 64, 1],
                [32, 128, 1],
            ]
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

    def gate_cnn(self, gate_cnn_input, gate_unit_config):
        """
        construct a gated cnn by the specific kernel size
        :param gate_unit_config:
        :param gate_cnn_input:
        :return:
        """
        conv1_out = Conv1D(gate_unit_config[0], gate_unit_config[1], strides=gate_unit_config[2])(gate_cnn_input)
        conv2_out = Conv1D(gate_unit_config[0], gate_unit_config[1], strides=gate_unit_config[2], activation="sigmoid")(
            gate_cnn_input)
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
        merged = None
        # add several ensemble gated cnn kernels
        for idx in range(len(self.summary['gate_units'])):
            if merged is None:
                merged = self.gate_cnn(embedding_out, self.summary['gate_units'][idx])
            else:
                merged = concatenate([merged, self.gate_cnn(embedding_out, self.summary['gate_units'][idx])])

        dense_output = Dense(128)(merged)
        dropout_out = Dropout(0.2)(dense_output)

        net_output = Dense(1, activation='sigmoid')(dropout_out)

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

        tensor_board = TensorBoard(log_dir='./logs/' + self.p_md5, batch_size=batch_size)
        early_stopping = EarlyStopping("val_loss", patience=self.get_p("e_s_patience"), verbose=0, mode='auto')
        file_path = "./models/" + self.p_md5 + "-{epoch:02d}-{val_acc:.2f}.h5"
        check_point = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [tensor_board, check_point]

        # Generators
        training_generator = DataGenerator(partition_train, self.train_df, self.label_df, batch_size)
        validation_generator = DataGenerator(partition_validation, self.train_df, self.label_df, batch_size)

        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        self.model.fit_generator(generator=training_generator,
                                 validation_data=validation_generator,
                                 use_multiprocessing=True,
                                 epochs=epochs,
                                 workers=6,
                                 callbacks=callbacks_list)
