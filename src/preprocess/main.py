import keras
from keras import Input
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Embedding, Conv1D, Multiply, GlobalMaxPooling1D, concatenate, Dropout
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn import metrics as sm

train = pd.read_csv("./data/train.csv", header=None, sep="|", names=['row_data'], error_bad_lines=False)
label = pd.read_csv("./data/train_label.csv", sep=",", error_bad_lines=False)
test = pd.read_csv("./data/test.csv", header=None, sep="|", names=['row_data'], error_bad_lines=False)