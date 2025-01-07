import os
import sys
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.layers import Input,Dense, Dropout, Flatten, Activation, Bidirectional, Concatenate
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.utils import to_categorical, plot_model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
K.set_learning_phase(False)
import shap
import shap.explainers.deep.deep_tf
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.utils import shuffle

#utilis
from utilis import readfile,one_hot_encode_along_row_axis,get_output,loc_to_model_loss


#Preparing the input data
#GLOBAL
#D:0,F:1,N:2,Q:3
label_dic = {0:'D',1:'F',2:'N',3:'Q'}
NUM_CLASSES = 4
selected_classes = np.array(list(range(NUM_CLASSES)))
SEQ_LEN = 500
SEQ_DIM = 4
seq_shape = (SEQ_LEN, SEQ_DIM)

#INPUT
data_folder = sys.argv[1]
model_folder = sys.argv[2]
DE_folder=sys.argv[3]
fa_type = sys.argv[4] ## Indicates which type of file the sequence appears in: train.fa, validation.fa, test.fa, or aim.fa
region_name = sys.argv[5] #the region name


train_filename = data_folder+'/train.fa'
valid_filename = data_folder+'/validation.fa'
test_filename = data_folder+'/test.fa'
aim_filename = 'aim.fa' # aim.fa is used to supplement newly identified sequences

filename = {'train':train_filename,'test':test_filename,'valid':valid_filename,'aim':aim_filename}

path_to_save_explainer = model_folder+'/explainer.pkl'

#OUTPUT
path_to_save_seq_pred = DE_folder+'/seq_pred.pkl'
path_to_save_shap_values = DE_folder+'/shap_values.pkl'
path_to_save_nuclei_contr = DE_folder+'/nuclei_contr.pkl'


#PROCESS
#get x_seq
ids, ids_d, seqs, classes = readfile(filename[fa_type]) # Generate the one-hot matrix for the 500 bp input sequences
x_seq=np.array(one_hot_encode_along_row_axis(seqs[region_name]))


#Loading model
model = loc_to_model_loss(model_folder)
conv_w = model.layers[3].get_weights()

#Performing prediction on one of the input sequences (x_seq)
prediction=model.predict(x_seq)[0]
with open(path_to_save_seq_pred, 'wb') as f:
	pickle.dump(prediction, f)
#plot
plt.plot(prediction,'--',marker="o")
plt.title(region_name)
plt.ylabel("Predictions")
plt.xlabel("Topics")
_ = plt.xticks(range(len(selected_classes)),[label_dic[i] for i in range(len(selected_classes))])
plt.savefig(DE_folder+'/seq_pred.svg', format='svg')


#in silico saturation mutagenesis on one of the input sequences (x_seq)
arrr_A = np.zeros((len(selected_classes),SEQ_LEN)) #(4,500)
arrr_C = np.zeros((len(selected_classes),SEQ_LEN)) #(4,500)
arrr_G = np.zeros((len(selected_classes),SEQ_LEN)) #(4,500)
arrr_T = np.zeros((len(selected_classes),SEQ_LEN)) #(4,500)
new_X = np.copy(x_seq)
real_score = prediction #model.predict(x_seq)[0], (None,4)

for mutloc in range(seq_shape[0]):
	new_X_= np.copy(new_X)
	new_X_[0][mutloc,:] = np.array([1, 0, 0, 0], dtype='int8')
	arrr_A[:,mutloc]=(real_score - model.predict(new_X_)[0])
	new_X_[0][mutloc,:] = np.array([0, 1, 0, 0], dtype='int8')
	arrr_C[:,mutloc]=(real_score - model.predict(new_X_)[0])
	new_X_[0][mutloc,:] = np.array([0, 0, 1, 0], dtype='int8')
	arrr_G[:,mutloc]=(real_score - model.predict(new_X_)[0])
	new_X_[0][mutloc,:] = np.array([0, 0, 0, 1], dtype='int8')
	arrr_T[:,mutloc]=(real_score - model.predict(new_X_)[0])
arrr_A[arrr_A==0]=None
arrr_C[arrr_C==0]=None
arrr_G[arrr_G==0]=None
arrr_T[arrr_T==0]=None
with open(path_to_save_nuclei_contr, 'wb') as f:
	pickle.dump(arrr_A, f)
	pickle.dump(arrr_C, f)
	pickle.dump(arrr_G, f)
	pickle.dump(arrr_T, f)


#DeepExplainer score on x_seq
#initialize DeepExplainer using train data
#train data
train_ids,train_ids_d, train_seqs, train_classes = readfile(train_filename)
X = np.array([one_hot_encode_along_row_axis(train_seqs[id]) for id in train_ids_d]).squeeze(axis=1) #(None,500,4)
ids = np.array([id for id in train_ids_d])
y = np.array([train_classes[id] for id in train_ids_d])
y = y[:,selected_classes]
X = X[y.sum(axis=1)>0]
ids = ids[y.sum(axis=1)>0]
y = y[y.sum(axis=1)>0]
#X_rc = [X,  X[:,::-1,::-1]]

#Loading model
model = loc_to_model_loss(model_folder)

#initialize
np.random.seed(seed=777)
rn=np.random.choice(X[0].shape[0], 500, replace=False)
explainer = shap.DeepExplainer((model.inputs, model.layers[-1].output), X[rn])

#DeepExplainer score
shap_values, indexes = explainer.shap_values(x_seq, ranked_outputs=1)
#len(shap_values) #list,len=1
#shap_values[0].shape #(1, 500, 4)
#len(indexes) #list,len=1
#indexes[0].shape #(1,)

with open(path_to_save_shap_values, 'wb') as f:
	pickle.dump(region_name, f)
	pickle.dump(x_seq, f) #x_seq.shape #(1, 500, 4)
	pickle.dump(shap_values, f) #shap_values[0].shape #(1, 500, 4)
	pickle.dump(indexes, f) #indexes[0].shape #(1,)




