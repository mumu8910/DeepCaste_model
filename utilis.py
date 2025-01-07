import sys
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

#GLOBAL
NUM_CLASSES = 4
selected_classes = np.array(list(range(NUM_CLASSES)))
SEQ_LEN = 500
SEQ_DIM = 4
seq_shape = (SEQ_LEN, SEQ_DIM)

#===========================================
#input
def readfile(filename): 
	global NUM_CLASSES
	ids = []
	ids_d = {}
	seqs = {}
	classes = {}
	f = open(filename, 'r')
	lines = f.readlines()
	f.close()
	seq = []
	for line in lines:
		if line[0] == '>':
			ids.append(line[1:].rstrip('\n'))
			name = line[1:].rstrip('\n').split('_')[0]
			label = line[1:].rstrip('\n').split('_')[1]
			if name not in seqs:
				seqs[name] = [] 
			if name not in ids_d:
				ids_d[name] = name
			if name not in classes:
				classes[name] = np.zeros(NUM_CLASSES)
			classes[name][int(label)] = 1
			if seq != []:
				seqs[ids[-2].split('_')[0]]= ("".join(seq))
			seq = []
		else:
			seq.append(line.rstrip('\n').upper())
	if seq != []:
		seqs[ids[-1].split('_')[0]]=("".join(seq))
	return ids,ids_d,seqs,classes

def readfile_wolabel(filename):
	ids = []
	ids_d = {}
	seqs = {}
	f = open(filename, 'r')
	lines = f.readlines()
	f.close()
	seq = []
	for line in lines:
		if line[0] == '>':
			name=line[1:].rstrip('\n')
			ids.append(name)
			if name not in seqs:
				seqs[name] = []
			if name not in ids_d:
				ids_d[name] = name
			if seq != []:
				seqs[ids[-2]]= ("".join(seq))
				seq = []
		else:
			seq.append(line.rstrip('\n').upper())
	if seq != []:
		seqs[ids[-1]]=("".join(seq))
	return ids,ids_d,seqs

def one_hot_encode_along_row_axis(sequence):
	to_return = np.zeros((1,len(sequence),4), dtype=np.int8)
	seq_to_one_hot_fill_in_array(zeros_array=to_return[0],sequence=sequence, one_hot_axis=1)
	return to_return

def seq_to_one_hot_fill_in_array(zeros_array, sequence, one_hot_axis):
	assert one_hot_axis==0 or one_hot_axis==1
	if (one_hot_axis==0):
		assert zeros_array.shape[1] == len(sequence)
	elif (one_hot_axis==1):
		assert zeros_array.shape[0] == len(sequence)
	for (i,char) in enumerate(sequence):
		if (char=="A" or char=="a"):
			char_idx = 0
		elif (char=="C" or char=="c"):
			char_idx = 1
		elif (char=="G" or char=="g"):
			char_idx = 2
		elif (char=="T" or char=="t"):
			char_idx = 3
		elif (char=="N" or char=="n"):
			continue
		else:
			raise RuntimeError("Unsupported character: "+str(char))
		if (one_hot_axis==0):
			zeros_array[char_idx,i] = 1
		elif (one_hot_axis==1):
			zeros_array[i,char_idx] = 1

#===========================================
#build model
def get_output(input_layer, hidden_layers):
	output = input_layer
	for hidden_layer in hidden_layers:
		output = hidden_layer(output)
	return output

#===========================================
#loss plot
def create_plots(history,model_folder):
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	#plt.savefig(model_folder + '/accuracy.png')
	plt.savefig(model_folder + '/accuracy.svg',format='svg')
	plt.clf()
	
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	#plt.savefig(model_folder + '/loss.png')
	plt.savefig(model_folder + '/loss.svg',format='svg')
	plt.clf()

#===========================================
##load model
def json_hdf5_to_model(json_filename, hdf5_filename):
	with open(json_filename, 'r') as f:
		model = model_from_json(f.read())
	model.load_weights(hdf5_filename)
	return model

def loc_to_model_loss(loc):
	return json_hdf5_to_model(loc + '/model.json', loc + '/model_best_loss.hdf5')

#===========================================
#perfomance
def shuffle_label(label):
	for i in range(len(label.T)):
		label.T[i] = shuffle(label.T[i])
	return label

def calculate_roc_pr(score, label):
	output = np.zeros((len(label.T), 2))
	for i in range(len(label.T)):
		roc_ = roc_auc_score(label.T[i], score.T[i])
		pr_ = average_precision_score(label.T[i], score.T[i])
		output[i] = [roc_, pr_]
	return output

#===========================================
#DeepExplainer plot
def plot_a(ax, base, left_edge, height, color):
	a_polygon_coords = [np.array([[0.0, 0.0],[0.5, 1.0],[0.5, 0.8],[0.2, 0.0],]),
	np.array([[1.0, 0.0],[0.5, 1.0],[0.5, 0.8],[0.8, 0.0],]),
	np.array([[0.225, 0.45],[0.775, 0.45],[0.85, 0.3],[0.15, 0.3],])]
	for polygon_coords in a_polygon_coords:
		ax.add_patch(matplotlib.patches.Polygon((np.array([1,height])[None,:]*polygon_coords+np.array([left_edge,base])[None,:]),facecolor=color, edgecolor=color))


def plot_c(ax, base, left_edge, height, color):
	ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,facecolor=color, edgecolor=color))
	ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,facecolor='white', edgecolor='white'))
	ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,facecolor='white', edgecolor='white', fill=True))


def plot_g(ax, base, left_edge, height, color):
	ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=1.3, height=height,facecolor=color, edgecolor=color))
	ax.add_patch(matplotlib.patches.Ellipse(xy=[left_edge+0.65, base+0.5*height], width=0.7*1.3, height=0.7*height,facecolor='white', edgecolor='white'))
	ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+1, base], width=1.0, height=height,facecolor='white', edgecolor='white', fill=True))
	ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.825, base+0.085*height], width=0.174, height=0.415*height,facecolor=color, edgecolor=color, fill=True))
	ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.625, base+0.35*height], width=0.374, height=0.15*height,facecolor=color, edgecolor=color, fill=True))

def plot_t(ax, base, left_edge, height, color):
	ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge+0.4, base],width=0.2, height=height, facecolor=color, edgecolor=color, fill=True))
	ax.add_patch(matplotlib.patches.Rectangle(xy=[left_edge, base+0.8*height],width=1.0, height=0.2*height, facecolor=color, edgecolor=color, fill=True))


#default
default_colors = {0:'green', 1:'blue', 2:'orange', 3:'red'}
default_plot_funcs = {0:plot_a, 1:plot_c, 2:plot_g, 3:plot_t}
def plot_weights_given_ax(ax, array,height_padding_factor,length_padding,subticks_frequency,highlight,colors=default_colors,plot_funcs=default_plot_funcs):	
	if len(array.shape)==3:
		array = np.squeeze(array)
	assert len(array.shape)==2, array.shape
	if (array.shape[0]==4 and array.shape[1] != 4):
		array = array.transpose(1,0)
	assert array.shape[1]==4
	
	max_pos_height = 0.0
	min_neg_height = 0.0
	heights_at_positions = []
	depths_at_positions = []
	for i in range(array.shape[0]):
		acgt_vals = sorted(enumerate(array[i,:]), key=lambda x: abs(x[1]))
		positive_height_so_far = 0.0
		negative_height_so_far = 0.0
		for letter in acgt_vals:
			plot_func = plot_funcs[letter[0]]
			color=colors[letter[0]]
			if (letter[1] > 0):
				height_so_far = positive_height_so_far
				positive_height_so_far += letter[1]
			else:
				height_so_far = negative_height_so_far
				negative_height_so_far += letter[1]
			plot_func(ax=ax, base=height_so_far, left_edge=i, height=letter[1],color=color)
		max_pos_height = max(max_pos_height, positive_height_so_far)
		min_neg_height = min(min_neg_height, negative_height_so_far)
		heights_at_positions.append(positive_height_so_far)
		depths_at_positions.append(negative_height_so_far)

	for color in highlight:
		start_pos=highlight[color][0]
		end_pos=highlight[color][1]
		assert start_pos >= 0.0 and end_pos <= array.shape[0]
		min_depth = np.min(depths_at_positions[start_pos:end_pos])
		max_height = np.max(heights_at_positions[start_pos:end_pos])
		ax.add_patch(matplotlib.patches.Rectangle(xy=[start_pos,min_depth],width=end_pos-start_pos,height=max_height-min_depth,edgecolor=color, fill=False))

	ax.set_xlim(-length_padding, array.shape[0]+length_padding)
	ax.xaxis.set_ticks(np.arange(0.0, array.shape[0]+1, subticks_frequency))
	height_padding = max(abs(min_neg_height)*(height_padding_factor),abs(max_pos_height)*(height_padding_factor))
	ax.set_ylim(min_neg_height-height_padding, max_pos_height+height_padding)
	return ax


def plot_weights_modified(array, fig, n,n1,n2, title='', ylab='',figsize=(20,2),height_padding_factor=0.2,length_padding=1.0,subticks_frequency=20,colors=default_colors,plot_funcs=default_plot_funcs,highlight={}):
	ax = fig.add_subplot(n,n1,n2)
	ax.set_title(title)
	ax.set_ylabel(ylab)
	y = plot_weights_given_ax(ax=ax, array=array,
	height_padding_factor=height_padding_factor,
	length_padding=length_padding,
	subticks_frequency=subticks_frequency,
	colors=colors,
	plot_funcs=plot_funcs,
	highlight=highlight)
	return fig,ax






