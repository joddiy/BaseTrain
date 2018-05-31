# -*- coding: utf-8 -*-
# file: t_auc.py
# author: joddiyzhang@gmail.com
# time: 2018/5/19 9:28 PM
# Copyright (C) <2017>  <Joddiy Zhang>
# ------------------------------------------------------------------------
import hashlib
import json
import time

import keras
from keras import layers as ll
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Input, Dense, Activation
from keras.layers.convolutional import Conv1D
from keras.layers.core import Flatten, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D, AveragePooling1D
from keras.models import load_model
from sklearn.model_selection import train_test_split

from src.config.config import *
from src.config.config import CACHE_DIR
from src.preprocess.pp_mal_conv import PPMalConv
from src.train.train import Train
from src.utils.data_generator_f import DataGeneratorF


class TResNet(Train):
    """
    train of mal conv
    """

    def __init__(self):
        self.train_df, self.label_df = PPMalConv().read_input()
        self.v_x = None
        self.v_y = None
        self.max_len = 1024
        self.history = None
        self.model = None
        self.p_md5 = None
        self.summary = {
            'input': DATA_CONFIG["mal_conv"]['train'],
            'time': time.time(),
            'batch_size': 32,
            'epochs': 16,
            's_test_size': 0.05,
            's_random_state': 1234,
            'e_s_patience': 2,
            'gate_units': [
                [128, 1, 1],
                # [128, 2, 1],
                # [128, 4, 1],
                # [128, 8, 1],
                [128, 16, 1],
                # [32, 32, 1],
                # [32, 64, 1],
                # [32, 128, 1],
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
        # conv2_out = Conv1D(gate_unit_config[0], gate_unit_config[1], strides=gate_unit_config[2], activation="sigmoid")(
        #     gate_cnn_input)
        # merged = Multiply()([conv1_out, conv2_out])
        gate_cnn_output = GlobalMaxPooling1D()(conv1_out)
        return gate_cnn_output

    def conv_block(self, input_tensor, kernel_size, filters, stage, block, strides=2):
        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = BatchNormalization(name=bn_name_base + '2a')(input_tensor)
        x = Activation('relu')(x)
        x = Conv1D(filters1, 1, strides=strides,
                   name=conv_name_base + '2a')(x)

        x = BatchNormalization(name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)
        x = Conv1D(filters2, kernel_size, padding='same',
                   name=conv_name_base + '2b')(x)

        x = BatchNormalization(name=bn_name_base + '2c')(x)
        x = Conv1D(filters3, 1, name=conv_name_base + '2c')(x)

        shortcut = BatchNormalization(name=bn_name_base + '1')(input_tensor)
        shortcut = Conv1D(filters3, 1, strides=strides,
                          name=conv_name_base + '1')(shortcut)

        x = ll.add([x, shortcut])
        x = Activation('relu')(x)
        return x

    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        filters1, filters2, filters3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = BatchNormalization(name=bn_name_base + '2a')(input_tensor)
        x = Activation('relu')(x)
        x = Conv1D(filters1, 1, name=conv_name_base + '2a')(x)

        x = BatchNormalization(name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)
        x = Conv1D(filters2, kernel_size,
                   padding='same', name=conv_name_base + '2b')(x)

        x = BatchNormalization(name=bn_name_base + '2c')(x)
        x = Conv1D(filters3, 1, name=conv_name_base + '2c')(x)

        x = ll.add([x, input_tensor])
        x = Activation('relu')(x)
        return x

    def resnet_block(self, input_tensor, final_layer_output=1, append='n'):
        x = Conv1D(
            64, 7, strides=2, padding='same', name='conv1' + append)(input_tensor)
        x = BatchNormalization(name='bn_conv1' + append)(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(3, strides=2)(x)
        x = self.conv_block(x, 3, [64, 64, 256],
                       stage=2, block='a' + append, strides=1)
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b' + append)
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c' + append)
        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a' + append)
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b' + append)
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c' + append)
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d' + append)
        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a' + append)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b' + append)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c' + append)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d' + append)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e' + append)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f' + append)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='g' + append)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='h' + append)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='i' + append)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='j' + append)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='k' + append)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='l' + append)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='m' + append)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='n' + append)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='o' + append)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='p' + append)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='q' + append)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='r' + append)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='s' + append)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='t' + append)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='u' + append)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='v' + append)
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='w' + append)
        x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a' + append)
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b' + append)
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c' + append)
        x = AveragePooling1D(final_layer_output, name='avg_pool' + append)(x)
        x = Flatten()(x)
        return x

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
        # # add several ensemble gated cnn kernels
        # for idx in range(len(self.summary['gate_units'])):
        #     if merged is None:
        #         merged = self.gate_cnn(embedding_out, self.summary['gate_units'][idx])
        #     else:
        #         merged = concatenate([merged, self.gate_cnn(embedding_out, self.summary['gate_units'][idx])])
        merged = self.resnet_block(embedding_out, 1, 'n')

        dense_output = Dense(128)(merged)
        dropout_out = Dropout(0.2)(dense_output)

        net_output = Dense(1, activation='sigmoid')(dropout_out)

        model = keras.models.Model(inputs=net_input, outputs=net_output)
        model.summary()

        return model

    def read_model(self, file_path):
        """
        read a model from file
        :return:
        """
        return load_model(file_path)

    def train(self):
        batch_size = self.get_p("batch_size")
        epochs = self.get_p("epochs")

        self.model = self.get_model()

        partition_train, partition_validation = train_test_split(range(len(self.train_df)), test_size=0.05)
        print('Length of the train: ', len(partition_train))
        print('Length of the validation: ', len(partition_validation))

        tensor_board = TensorBoard(log_dir='./logs/' + self.p_md5, batch_size=batch_size)
        early_stopping = EarlyStopping("val_loss", patience=self.get_p("e_s_patience"), verbose=0, mode='auto')
        file_path = "./models/" + self.p_md5 + "-{epoch:04d}-{val_loss:.5f}-{val_acc:.5f}.h5"
        check_point = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        callbacks_list = [tensor_board, check_point]

        # Generators
        training_generator = DataGeneratorF(partition_train, self.train_df, self.label_df, batch_size, dim=self.max_len)
        validation_generator = DataGeneratorF(partition_validation, self.train_df, self.label_df, batch_size,
                                              dim=self.max_len)

        self.model.compile(loss='binary_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy'])

        self.model.fit_generator(generator=training_generator,
                                 validation_data=validation_generator,
                                 use_multiprocessing=True,
                                 epochs=epochs,
                                 workers=6,
                                 callbacks=callbacks_list)
