import os
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib
import matplotlib.pyplot as plt

#utilis
from utilis import plot_weights_modified

#GLOBAL
#D:0,F:1,N:2,Q:3
NUM_CLASSES = 4
selected_classes = np.array(list(range(NUM_CLASSES)))
label_dic = {0:'D',1:'F',2:'N',3:'Q'}
SEQ_LEN = 500
SEQ_DIM = 4
seq_shape = (SEQ_LEN, SEQ_DIM)

#INPUT
DE_folder=sys.argv[1]

path_to_save_shap_values = DE_folder+'/shap_values.pkl'
path_to_save_nuclei_contr = DE_folder+'/nuclei_contr.pkl'
path_to_save_high_light = DE_folder+'/high_light.txt'

#subplot
ntrack=1+len(selected_classes)
fig = plt.figure(figsize=(65,25))

##Plotting DeepExplainer score on one of the input sequences
#load DeepExplainer score on x_seq
with open(path_to_save_shap_values, 'rb') as f:
	region_name = pickle.load(f)
	x_seq = pickle.load(f) #x_seq.shape #(1, 500, 4)
	shap_values = pickle.load(f) #shap_values[0].shape #(1, 500, 4)
	indexes = pickle.load(f)  #indexes[0].shape #(1,)

#motif hit
highlight={}
with open(path_to_save_high_light, 'r') as inp:
	line = inp.readline()
	while line != '':
		m = line.strip('\n').split('\t')
		highlight[m[0]] = [int(m[1]), int(m[2])]
		line = inp.readline()
#plot
_, ax1 =plot_weights_modified(shap_values[0][0]*x_seq,fig,ntrack,1,1,title=region_name, subticks_frequency=10,highlight=highlight,ylab="DeepExplainer")

#==============

#in silico saturation mutagenesis on one of the input sequences
#load
with open(path_to_save_nuclei_contr, 'rb') as f:
    arrr_A=pickle.load(f)
    arrr_C=pickle.load(f)
    arrr_G=pickle.load(f)
    arrr_T=pickle.load(f)

#plot
for i in range(len(selected_classes)):
	group=selected_classes[i]
	ax = fig.add_subplot(ntrack,1,i+2)
	ax.set_title(label_dic[group])
	ax.scatter(range(seq_shape[0]),-1*arrr_A[group],label='A',color='green')
	ax.scatter(range(seq_shape[0]),-1*arrr_C[group],label='C',color='blue')
	ax.scatter(range(seq_shape[0]),-1*arrr_G[group],label='G',color='orange')
	ax.scatter(range(seq_shape[0]),-1*arrr_T[group],label='T',color='red')
	ax.legend()
	ax.axhline(y=0,linestyle='--',color='gray')
	ax.set_xlim((0,seq_shape[0]))
	_ = ax.set_xticks(np.arange(0, seq_shape[0]+1, 10))

#plt.savefig(DE_folder+'/seq.png')
plt.savefig(DE_folder+'/seq.svg', format='svg')


