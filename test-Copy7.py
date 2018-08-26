
# coding: utf-8

# ## Read data from database

# In[1]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


# In[2]:


import pymysql
from warnings import filterwarnings

_connection = None

def get_connection(db_config):
    """
    get db connection
    :return:
    """
    global _connection
    if _connection is None:
        _connection = pymysql.connect(host=db_config['host'], user=db_config['username'],
                                      password=db_config['password'],
                                      db=db_config['db'], charset="utf8")
        filterwarnings('ignore', category=pymysql.Warning)

    return _connection


def close():
    """
    close DB connection
    :return:
    """
    global _connection
    if _connection is not None:
        _connection.close()
    _connection = None


# In[3]:


db = {
    'host': 'ncrs.d2.comp.nus.edu.sg',
    'username': 'malware_r',
    'password': 'GEg22v2O7jbfWhb3',
    'db': 'malware'
}


# In[4]:


# the base function which can query sql and return dict data
def get_specific_data(table_suffix, sql=None):
    start_time = time.time()
    
    global _connection
    if _connection is None:
        raise Exception("please init db connect first")

    cursor = _connection.cursor()
    cursor.execute("SET NAMES utf8mb4")

    ret = []
        
    cursor.execute(sql)

    field_names = [i[0] for i in cursor.description]

    for row in cursor:
        temp = {}
        for key in range(len(row)):
            temp[field_names[key]] = row[key]
        ret.append(temp)
     
    cursor.close()
    # _connection.close()
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return ret


# In[5]:


close()
res1 = []
get_connection(db)
table_suffix = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E"]
# Iterate all partitions of databases
for suffix in table_suffix:
    sql = """ 
select
  a.mw_file_hash,
  a.section_name,
  c.mw_file_suffix as mw_file_size,
  c.mw_file_prefix as mw_file_directory,
  c.mw_num_engines,
  a.pointerto_raw_data,
  a.virtual_size,
  d.mw_em_f
from mw_index_2017_section_%s as a
  inner join mw_index_2017_%s c on a.mw_file_hash = c.mw_file_hash
  inner join mw_index_2017_feature_%s d on a.mw_file_hash = d.mw_file_hash
where (CNT_CODE = 1 or MEM_EXECUTE = 1) and c.mw_num_engines <> -1 and (c.mw_num_engines >= 4 or c.mw_num_engines = 0) and
      c.mw_file_prefix in ('201704')
    """ % (suffix, suffix, suffix)
    res1.extend(get_specific_data(suffix, sql))
close()
print(len(res1))


# In[6]:


close()
res2 = []
get_connection(db)
table_suffix = ["F"]
# Iterate all partitions of databases
for suffix in table_suffix:
    sql = """ 
select
  a.mw_file_hash,
  a.section_name,
  c.mw_file_suffix as mw_file_size,
  c.mw_file_prefix as mw_file_directory,
  c.mw_num_engines,
  a.pointerto_raw_data,
  a.virtual_size,
  d.mw_em_f
from mw_index_2017_section_%s as a
  inner join mw_index_2017_%s c on a.mw_file_hash = c.mw_file_hash
  inner join mw_index_2017_feature_%s d on a.mw_file_hash = d.mw_file_hash
where (CNT_CODE = 1 or MEM_EXECUTE = 1) and c.mw_num_engines <> -1 and (c.mw_num_engines >= 4 or c.mw_num_engines = 0) and
      c.mw_file_prefix in ('201704')
    """ % (suffix, suffix, suffix)
    res2.extend(get_specific_data(suffix, sql))
close()
print(len(res2))


# ## Check and split data

# In[8]:


import pandas as pd

x_train = pd.DataFrame(res1)
x_train.mw_num_engines[x_train.mw_num_engines == 0 ] = 0
x_train.mw_num_engines[x_train.mw_num_engines >= 4 ] = 1
y_train = x_train.mw_num_engines.ravel()

x_val = pd.DataFrame(res2)
x_val.mw_num_engines[x_val.mw_num_engines == 0 ] = 0
x_val.mw_num_engines[x_val.mw_num_engines >= 4 ] = 1
y_val = x_val.mw_num_engines.ravel()

# ## Autoencoder

# In[11]:


from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


# In[34]:


import keras
import numpy as np

max_length = 168070


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, datasets, batch_size=32, dim=max_length, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
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
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, self.dim), dtype=float)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            base_path = "/ssd/2017/{0}/{1}{2}"
            item = self.datasets.loc[ID]
            file_path = base_path.format(item["mw_file_directory"], item["mw_file_hash"], item["mw_file_size"])
            in_file = open(file_path, 'rb')
            in_file.seek(item['pointerto_raw_data'])
            if item['virtual_size'] > self.dim:
                bytes_data = [int(single_byte) for single_byte in in_file.read(self.dim)]
            else:
                bytes_data = [int(single_byte) for single_byte in in_file.read(item['virtual_size'])]
            X[i, 0:len(bytes_data)] = bytes_data

#         X = X.reshape((-1, 100, 100, 1)) / 255.0
#         T = X.reshape((-1, 10000)) / 255.0
        Y = X.reshape((-1, max_length, 1)) / 255.0
        return Y, Y


# In[39]:


import time
from keras import Input
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Dense, Embedding, Conv1D, Conv2D, Multiply, GlobalMaxPooling1D, Dropout, Activation
from keras.layers import UpSampling2D, Flatten, merge, MaxPooling2D, MaxPooling1D, UpSampling1D
from keras.models import load_model, Model
from keras.layers import Dropout, BatchNormalization, Maximum, Add, concatenate
from keras.optimizers import RMSprop

class Autoencoder():
    def __init__(self):
        self.autoencoder = None
        self.encoder = None
        self.start_time = time.time()

    def get_model(self):

        inputs = Input(shape=(max_length, 1));
        
        conv1 = Conv1D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv1D(8, 3, activation = 'relu', padding = 'causal', kernel_initializer = 'he_normal', dilation_rate=16)(conv1)
        pool1 = MaxPooling1D(7)(conv1)
        
        conv2 = Conv1D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv1D(16, 3, activation = 'relu', padding = 'causal', kernel_initializer = 'he_normal', dilation_rate=8)(conv2)
        pool2 = MaxPooling1D(7)(conv2)
        
        conv3  = Conv1D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv1D(32, 3, activation = 'relu', padding = 'causal', kernel_initializer = 'he_normal', dilation_rate=4)(conv3)
        pool3 = MaxPooling1D(7)(conv3)
        
        conv4 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv1D(64, 3, activation = 'relu', padding = 'causal', kernel_initializer = 'he_normal', dilation_rate=2)(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling1D(7)(drop4)
        
        conv5 = Conv1D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv1D(128, 3, activation = 'relu', padding = 'causal', kernel_initializer = 'he_normal', dilation_rate=1)(conv5)
        drop5 = Dropout(0.5)(conv5)
        
        encoded = Conv1D(1, 1, activation = 'sigmoid')(drop5)
        
        up6 = Conv1D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(7)(encoded))
        merge6 = concatenate([drop4, up6])
        conv6 = Conv1D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv1D(64, 3, activation = 'relu', padding = 'causal', kernel_initializer = 'he_normal', dilation_rate=2)(conv6)
        
        up7 = Conv1D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(7)(conv6))
        merge7 = concatenate([conv3, up7])
        conv7 = Conv1D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv1D(32, 3, activation = 'relu', padding = 'causal', kernel_initializer = 'he_normal', dilation_rate=4)(conv7)
        
        up8 = Conv1D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(7)(conv7))
        merge8 = concatenate([conv2, up8])
        conv8 = Conv1D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv1D(16, 3, activation = 'relu', padding = 'causal', kernel_initializer = 'he_normal', dilation_rate=8)(conv8)
        
        up9 = Conv1D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling1D(7)(conv8))
        merge9 = concatenate([conv1, up9])
        conv9 = Conv1D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv1D(8, 3, activation = 'relu', padding = 'causal', kernel_initializer = 'he_normal', dilation_rate=16)(conv9)
        conv9 = Conv1D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv1D(1, 1, activation = 'sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10)

        model.compile(loss='mean_squared_error', optimizer=RMSprop())
        model.summary()
        return model

    def train(self, max_epoch, batch_size=32):

        model = self.get_model()
        
        print('Length of the train: ', len(x_train))
        print('Length of the validation: ', len(x_val))

        tensor_board = TensorBoard(log_dir='/home/zhaoqi/autoencoder/log/', batch_size=batch_size)
        file_path = "/home/zhaoqi/autoencoder/models/"+ str(self.start_time) +"-{epoch:04d}-{val_loss:.5f}.h5"
        #         early_stopping = EarlyStopping("val_loss", patience=2, verbose=0, mode='auto')
        check_point = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        callbacks_list = [check_point, tensor_board]
        
        # Generators
        training_generator = DataGenerator(range(len(x_train)), x_train, batch_size)
        validation_generator = DataGenerator(range(len(x_val)), x_val, batch_size)

        model.fit_generator(generator=training_generator,
                                       validation_data=validation_generator,
                                       use_multiprocessing=True,
                                        epochs=max_epoch,
                                       workers=6,
                                       callbacks=callbacks_list)
        
autoencoder = Autoencoder()
autoencoder.train(max_epoch=512, batch_size=16)