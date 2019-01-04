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
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Concatenate
from keras import backend as K
from keras.utils import plot_model
from iter import *
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from importlib import import_module
import matplotlib.pyplot as plt
plt.switch_backend('agg')
config =tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.2
set_session(tf.Session(config=config))

batch_size = 256

parser=argparse.ArgumentParser()
parser.add_argument("--rat",type=float,default=0.6,help='ratio for weak qg batch')
parser.add_argument("--end",type=float,default=0.3,help='end ratio')
parser.add_argument("--epoch",type=int,default=3,help='epoch')
parser.add_argument("--save1",type=str,default="gooasym1",help='rch')
parser.add_argument("--save2",type=str,default="googru1",help='rch')
parser.add_argument("--rc",type=str,default='rc',help='rnn or cnn')
args=parser.parse_args()

# input image dimensions
img_rows, img_cols = 33, 33

# the data, shuffled and split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

#if K.image_data_format() == 'channels_first':
#    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#    input_shape = (1, img_rows, img_cols)
#else:
#x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape1= (3,33,33)
input_shape2= (20,4)

#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.Adadelta(),
#              metrics=['accuracy'])
#model=keras.models.load_model('save/fullydijetsame_10')
savename="save/"+str(args.save1)
epoch=eval(open(savename+"/history").readline())+1
model1=keras.models.load_model(savename+"/check_"+str(epoch))
for i in range(len(model1.layers)):
  model1.layers[i].name+="_1"
savename="save/"+str(args.save2)
epoch=eval(open(savename+"/history").readline())+1
model2=keras.models.load_model(savename+"/check_"+str(epoch))
for i in range(len(model2.layers)):
  model2.layers[i].name+="_2"

concat=Concatenate(axis=1,name='concat')([model1.get_layer(index=len(model1.layers)-2).output,model2.get_layer(index=len(model2.layers)-2).output])
dens=Dense(2048,activation='relu',name='dens')(concat)
drop=Dropout(0.5,name='drop')(dens)
out=Dense(2,activation='softmax',name='out')(drop)
imodel=Model(inputs=[model1.input,model2.input],outputs=out)

rc=""
for sha in imodel._feed_inputs:
  if(len(sha._keras_shape)==4):
    rc+="c"
  if(len(sha._keras_shape)==3):
    rc+="r"
train=wkiter(["/scratch/yjdata/quark100_img.root","/scratch/yjdata/gluon100_img.root"],batch_size=batch_size,end=args.end*3./5.,istrain=1,rc=rc)
valid=wkiter(["/scratch/yjdata/quark100_img.root","/scratch/yjdata/gluon100_img.root"],batch_size=batch_size,begin=4./5.,end=args.end*1./5.+4./5.,rc=rc)

imodel.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.SGD(),metrics=['accuracy'])
plot_model(imodel,to_file='imodel.png')

imodel.fit_generator(train.next(),steps_per_epoch=train.totalnum(),validation_data=valid.next(),validation_steps=valid.totalnum(),epochs=args.epoch,verbose=1)

