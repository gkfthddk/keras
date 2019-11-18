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
from keras.layers import Dense, Dropout, Concatenate, Average
from keras import backend as K
from keras.utils import plot_model
from siter import *
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
parser.add_argument("--epoch",type=int,default=1,help='epoch')
parser.add_argument("--save1",type=str,default="gooasym1",help='rch')
parser.add_argument("--save2",type=str,default="googru1",help='rch')
parser.add_argument("--left",type=str,default="/scratch/yjdata/quark100_img",help='which train sample (qq,gg,zq,zg)')
parser.add_argument("--right",type=str,default="/scratch/yjdata/gluon100_img",help='which train sample (qq,gg,zq,zg)')
parser.add_argument("--valleft",type=str,default="/scratch/yjdata/quark100_img",help='which train sample (qq,gg,zq,zg)')
parser.add_argument("--valright",type=str,default="/scratch/yjdata/gluon100_img",help='which train sample (qq,gg,zq,zg)')
parser.add_argument("--rc",type=str,default='rc',help='rnn or cnn')
args=parser.parse_args()

# input image dimensions
img_rows, img_cols = 33, 33

input_shape1= (3,33,33)
input_shape2= (20,4)

#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.Adadelta(),
#              metrics=['accuracy'])
#model=keras.models.load_model('save/fullydijetsame_10')
equa=''
savename="save/"+str(args.save1)
history=open(savename+'/history')
epoch=eval(history.readline())+1
acc=eval(history.readline())['val_acc'][epoch-1]
history.close()
equa+="{}\t{:.3f}".format(args.save1,acc)
model1=keras.models.load_model(savename+"/check_"+str(epoch))
for i in range(len(model1.layers)):
  model1.layers[i].name+="_1"
savename="save/"+str(args.save2)
history=open(savename+'/history')
epoch=eval(history.readline())+1
acc=eval(history.readline())['val_acc'][epoch-1]
history.close()
equa+="{}\t{:.3f}".format(args.save2,acc)
model2=keras.models.load_model(savename+"/check_"+str(epoch))
for i in range(len(model2.layers)):
  model2.layers[i].name+="_2"

out=Average()([model1.outputs[0],model2.outputs[0]])
imodel=Model(inputs=[model1.input,model2.input],outputs=out,name='ensemble')
#imodel=Model(inputs=model2.input,outputs=model2.outputs[0],name='ensemble')

rc=""
for sha in imodel._feed_inputs:
  if(len(sha._keras_shape)==4):
    rc+="c"
  if(len(sha._keras_shape)==3):
    rc+="r"
train=wkiter([args.left+".root",args.right+".root"],batch_size=batch_size,end=args.end*1.,istrain=1,rc=rc)
valid=wkiter([args.valleft+".root",args.valright+".root"],batch_size=batch_size,end=args.end*1.,rc=rc)

imodel.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.SGD(lr=0.0000001),metrics=['accuracy'])
plot_model(imodel,to_file='imodel.png')

imodel.fit_generator(train.next(),1,validation_data=valid.next(),validation_steps=valid.totalnum(),epochs=args.epoch,verbose=1)

print(equa)
