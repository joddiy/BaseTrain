# coding: utf-8

# In[1]:


import pandas as pd
from keras import Input
from keras.layers import Dense, Embedding, Conv1D, Multiply, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split


# In[2]:


def gate_cnn(gate_cnn_input):
    """
    construct a gated cnn by the specific kernel size
    :param gate_cnn_input:
    :param kernel_size:
    :return:
    """
    conv1_out = Conv1D(128, 500, strides=1)(gate_cnn_input)
    conv2_out = Conv1D(128, 500, strides=1, activation="sigmoid")(gate_cnn_input)
    merged = Multiply()([conv1_out, conv2_out])
    gate_cnn_output = GlobalMaxPooling1D()(merged)
    return gate_cnn_output


# In[3]:


def get_model():
    """
    get a model
    :param max_len:
    :param kernel_sizes:
    :return:
    """
    net_input = Input(shape=(8192,))

    embedding_out = Embedding(256, 8, input_length=8192)(net_input)
    merged = gate_cnn(embedding_out)

    dense_out = Dense(128)(merged)
    net_output = Dense(1, activation='sigmoid')(dense_out)

    model = keras.models.Model(inputs=net_input, outputs=net_output)
    model.summary()

    return model


# In[4]:


import keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, datasets, labels, batch_size=32, dim=8192, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.datasets = datasets
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def crop_exceed_data(self, data):
        if len(data) <= 8192:
            return data
        return data[0: 8192]

    def get_bytes_array(self, data):
        """
        int to bytes array
        :param data:
        :return:
        """
        bytes_data = bytes(map(int, data.split(",")))
        bytes_data = self.crop_exceed_data(bytes_data)
        return [int(single_byte) for single_byte in bytes_data]

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, self.dim), dtype=int)
        y = np.zeros(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            bytes_data = self.get_bytes_array(self.datasets[ID])
            X[i, 0:len(bytes_data)] = bytes_data
            # Store class
            y[i] = self.labels[ID]

        return X, y


# In[5]:


from os import listdir
from os.path import isfile, join

datasets = []
labels = []
input_dir = '/hdd1/malware_data/'
# train data
files = [input_dir + f for f in listdir(input_dir) if isfile(join(input_dir, f)) and f[-9:] == 'train.csv']
for file_name in files:
    datasets.append(pd.read_csv(file_name, header=None, sep="|", index_col=None))
datasets = pd.concat(datasets, ignore_index=True)[0]

# train label
files = [input_dir + f for f in listdir(input_dir) if isfile(join(input_dir, f)) and f[-15:] == 'train_label.csv']
for file_name in files:
    labels.append(pd.read_csv(file_name, header=None, index_col=None))
labels = pd.concat(labels, ignore_index=True)[0]

print('Length of the data: ', len(datasets))

# In[6]:


import numpy as np
from keras.callbacks import TensorBoard
import time

batch_size = 32

partition_train, partition_validation = train_test_split(range(len(datasets)), test_size=0.05)
print('Length of the train: ', len(partition_train))
print('Length of the validation: ', len(partition_validation))

# Generators
training_generator = DataGenerator(partition_train, datasets, labels, batch_size)
validation_generator = DataGenerator(partition_validation, datasets, labels, batch_size)

tensorboard = TensorBoard(log_dir='./logs/{}'.format(time.time()), batch_size=batch_size)

model = get_model()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    epochs=9,
                    workers=6,
                    callbacks=[tensorboard])

model.save('./models/{}.h5'.format(time.time()))
