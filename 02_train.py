import sys
import os
import optparse
from array import *
import tensorflow
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
from utilis import readfile,one_hot_encode_along_row_axis,get_output,create_plots

#build model
def build_model():
	#global
	global seq_shape
	#fun
	reverse_lambda_ax2 = Lambda(lambda x: K.reverse(x,axes=2))
	reverse_lambda_ax1 = Lambda(lambda x: K.reverse(x,axes=1))
	#input
	forward_input = Input(shape=seq_shape)
	reverse_input = reverse_lambda_ax2(reverse_lambda_ax1(forward_input))
	#hidden layers
	layer0 = [
	Conv1D(128, kernel_size=20, padding="valid", activation='relu', kernel_initializer='random_uniform'),
	MaxPooling1D(pool_size=10, strides=10, padding='valid'),
	Dropout(0.2),
	TimeDistributed(Dense(128, activation='relu')),
	Bidirectional(LSTM(128, dropout=0.1, recurrent_dropout=0.1, return_sequences=True)),
	Dropout(0.2),
	Flatten(),
	Dense(64, activation='relu'),
	Dropout(0.4),
	Dense(4, activation='sigmoid')]
	#output
	forward_output = get_output(forward_input, layer0)
	reverse_output = get_output(reverse_input, layer0)
	output = average([forward_output,reverse_output])
	#model
	model = Model(input=forward_input, output=output)
	model.summary()
	adam = Adam(lr=0.001)
	model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
	return model

#============================================

#Preparing the input data
#GLOBAL
#D:0,F:1,N:2,Q:3
NUM_CLASSES = 4
selected_classes = np.array(list(range(NUM_CLASSES)))
SEQ_LEN = 500
SEQ_DIM = 4
seq_shape = (SEQ_LEN, SEQ_DIM)

BATCH = 128
EPOCH = 32

#INPUT
data_folder = sys.argv[1]
model_folder = sys.argv[2]
#test
train_filename = data_folder+'/train.fa'
valid_filename = data_folder+'/validation.fa'
test_filename = data_folder+'/test.fa'

#OUTPUT
path_to_save_arc = model_folder + '/model.json'
path_to_save_end_weights = model_folder + '/model_best_loss.hdf5'
path_to_save_history = model_folder + '/history.pkl'

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

#Training the model
print("Compile model...")
model = build_model()

model_json = model.to_json()
with open(path_to_save_arc, "w") as json_file:
    json_file.write(model_json)
print("Model architecture saved to..", path_to_save_arc)

print("Train model...")
history = model.fit(train_data, y_train, nb_epoch=EPOCH, batch_size=BATCH, shuffle=True, validation_data=(valid_data, y_valid), verbose=1)
with open(path_to_save_history,'wb') as f:
        pickle.dump(history, f)
create_plots(history,model_folder)

model.save_weights(path_to_save_end_weights)
print("Model weights saved to..", path_to_save_end_weights)

plot_model(model, to_file=model_folder + '/model.png')

score, acc = model.evaluate(test_data, y_test, batch_size=BATCH)
print('Test score:', score)
print('Test accuracy:', acc)





