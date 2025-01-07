import sys
import os
import optparse
from array import *
import tensorflow as tf
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.utils import class_weight, shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Bidirectional, Concatenate, PReLU
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers import Layer, average, Input, Lambda, concatenate
from keras.models import Model
from keras.utils import plot_model
import keras.backend as K



#utilis
from utilis import readfile,one_hot_encode_along_row_axis,loc_to_model_loss,shuffle_label,calculate_roc_pr


#Preparing the input data
#GLOBAL
#D:0,F:1,N:2,Q:3
NUM_CLASSES = 4
selected_classes = np.array(list(range(NUM_CLASSES)))
SEQ_LEN = 500
SEQ_DIM = 4
seq_shape = (SEQ_LEN, SEQ_DIM)

#INPUT
data_folder = sys.argv[1]
model_folder = sys.argv[2]

train_filename = data_folder+'/train.fa'
valid_filename = data_folder+'/validation.fa'
test_filename = data_folder+'/test.fa'

#OUTPUT
path_to_save_performance = model_folder + '/performance.pkl'

#PROCESS
print("Prepare input...")
#train
train_ids, train_ids_d, train_seqs, train_classes = readfile(train_filename)
X_train = np.array([one_hot_encode_along_row_axis(train_seqs[id]) for id in train_ids_d]).squeeze(axis=1)
y_train = np.array([train_classes[id] for id in train_ids_d])
y_train = y_train[:,selected_classes]
X_train = X_train[y_train.sum(axis=1)>0]
y_train = y_train[y_train.sum(axis=1)>0]
train_data = X_train
#validation
valid_ids, valid_ids_d, valid_seqs, valid_classes = readfile(valid_filename)
X_valid = np.array([one_hot_encode_along_row_axis(valid_seqs[id]) for id in valid_ids_d]).squeeze(axis=1)
y_valid = np.array([valid_classes[id] for id in valid_ids_d])
y_valid = y_valid[:,selected_classes]
X_valid = X_valid[y_valid.sum(axis=1)>0]
y_valid = y_valid[y_valid.sum(axis=1)>0]
valid_data = X_valid
#test
test_ids, test_ids_d, test_seqs, test_classes = readfile(test_filename)
X_test = np.array([one_hot_encode_along_row_axis(test_seqs[id]) for id in test_ids_d]).squeeze(axis=1)
y_test = np.array([test_classes[id] for id in test_ids_d])
y_test = y_test[:,selected_classes]
X_test = X_test[y_test.sum(axis=1)>0]
y_test = y_test[y_test.sum(axis=1)>0]
test_data = X_test

#Loading model
model = loc_to_model_loss(model_folder)

#Calculating auROC and auPR values
print('calculate roc and pr...')
roc_pr_dict = {"train": {}, "valid":{},"test": {}, "shuffle": {}}
roc_pr_dict["train"]["score"] = model.predict(train_data)
roc_pr_dict["train"]["label"] = y_train
roc_pr_dict["valid"]["score"] = model.predict(valid_data)
roc_pr_dict["valid"]["label"] = y_valid
roc_pr_dict["test"]["score"] = model.predict(test_data)
roc_pr_dict["test"]["label"] = y_test
roc_pr_dict["shuffle"]["score"] = np.array(roc_pr_dict["train"]["score"], copy=True)
roc_pr_dict["shuffle"]["label"] = shuffle_label(np.array(y_train, copy=True))

for sets in ["train", "valid", "test", "shuffle"]:
	roc_pr_dict[sets]["roc_pr"] = calculate_roc_pr(roc_pr_dict[sets]["score"], roc_pr_dict[sets]["label"])
with open(path_to_save_performance,'wb') as f:
        pickle.dump(roc_pr_dict, f)

#plot
fig = plt.figure(figsize=(8, 10))
ax = fig.add_subplot(2, 1, 1)
ax.set_ylabel('auROC')
ax.scatter(selected_classes, roc_pr_dict["train"]["roc_pr"].T[0], color='red', label='TRAIN')
ax.scatter(selected_classes, roc_pr_dict["valid"]["roc_pr"].T[0], color='green', label='VALID')
ax.scatter(selected_classes, roc_pr_dict["test"]["roc_pr"].T[0], color='blue', label='TEST')
ax.scatter(selected_classes, roc_pr_dict["shuffle"]["roc_pr"].T[0], color='gray', label='SHUFFLED')
ax.set_ylim([0, 1])
_ = plt.xticks(range(len(selected_classes)),range(1,len(selected_classes)+1))
ax.legend()

ax = fig.add_subplot(2, 1, 2)
ax.set_ylabel('auPR')
ax.scatter(selected_classes, roc_pr_dict["train"]["roc_pr"].T[1], color='red', label='TRAIN')
ax.scatter(selected_classes, roc_pr_dict["valid"]["roc_pr"].T[1], color='green', label='VALID')
ax.scatter(selected_classes, roc_pr_dict["test"]["roc_pr"].T[1], color='blue', label='TEST')
ax.scatter(selected_classes, roc_pr_dict["shuffle"]["roc_pr"].T[1], color='gray', label='SHUFFLED')
ax.set_ylim([0, 1])
_ = plt.xticks(range(len(selected_classes)),range(1,len(selected_classes)+1))

#plt.savefig(model_folder +'/performance.png')
plt.savefig(model_folder +'/performance.svg', format='svg')














