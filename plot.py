'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
import argparse
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Embedding
from keras.layers import Conv2D, MaxPooling2D, SimpleRNN
from keras.utils import plot_model
from keras import backend as K
from iter import *
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from importlib import import_module
import datetime
start=datetime.datetime.now()
config =tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.4
set_session(tf.Session(config=config))

def weakloss(ytrue,ypred):
	a=K.sum(ypred)/ypred.shape[0] 
	b=K.sum(ytrue)/ypred.shape[0]
	loss=a - b
	print(type(loss))
	loss=K.square(loss)
	return loss
def mean_squared_error(y_true,y_pred):
	return K.mean(K.square(y_pred - y_true),axis=-1)
batch_size = 128
num_classes = 2
epochs = 10 

parser=argparse.ArgumentParser()
parser.add_argument("--network",type=str,default="lstm",help='network name on symbols/')
parser.add_argument("--loss",type=str,default="binary_crossentropy",help='network name on symbols/')
args=parser.parse_args()

# input image dimensions
img_rows, img_cols = 33, 33

input_shape = (20,4)
im_shape=(3,33,33)
imnet=import_module('symbols.asym')
if(args.loss=="weakloss"):args.loss=weakloss
net=import_module('symbols.'+args.network)
model=net.get_symbol(input_shape,num_classes)
model2=imnet.get_symbol(im_shape,num_classes)
plot_model(model,to_file='nplot.png')
plot_model(model2,to_file='implot.png')
print (datetime.datetime.now()-start)
