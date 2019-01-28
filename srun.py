'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--rat",type=float,default=0.6,help='ratio for weak qg batch')
parser.add_argument("--end",type=float,default=1.,help='end ratio')
parser.add_argument("--save",type=str,default="test_",help='save name')
parser.add_argument("--network",type=str,default="rnncnn",help='network name on symbols/')
parser.add_argument("--right",type=str,default="/scratch/yjdata/gluon100_img",help='which train sample (qq,gg,zq,zg)')
parser.add_argument("--pt",type=int,default=100,help='pt range pt~pt*1.1')
parser.add_argument("--ztest",type=int,default=0,help='true get zjet test')
parser.add_argument("--epochs",type=int,default=20,help='num epochs')
parser.add_argument("--loss",type=str,default="categorical_crossentropy",help='network name on symbols/')
parser.add_argument("--gpu",type=int,default=0,help='gpu number')
args=parser.parse_args()
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu) 
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Embedding
from keras.layers import Conv2D, MaxPooling2D, SimpleRNN
from keras import backend as K
from keras.utils import plot_model
from titer import *
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from importlib import import_module
import datetime
start=datetime.datetime.now()
config =tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.4
set_session(tf.Session(config=config))

batch_size = 256
num_classes = 2 


epochs = args.epochs
print(epochs)

# input image dimensions
img_rows, img_cols = 33, 33

input_shape1 = (10,33,33)
input_shape2 = (20,9)

if(args.loss=="weakloss"):args.loss=weakloss
net=import_module('symbols.'+args.network)
try:
  onehot=net.onehot()
  input_shape2=(20,5)
except:onehot=0
rc=net.rc()
print(rc)
if(rc=="rc"):model=net.get_symbol(input_shape1,input_shape2)
if(rc=="r"):model=net.get_symbol(input_shape2)
if(rc=="c"):model=net.get_symbol(input_shape1)
rc=""
for sha in model._feed_inputs:
  if(len(sha._keras_shape)==4):
    rc+="c"
  if(len(sha._keras_shape)==3):
    rc+="r"

print("### plot done ###")
#model.compile(loss='mean_squared_error',
model.compile(loss=args.loss,
              optimizer=keras.optimizers.SGD(),
	      metrics=['accuracy'])
"""model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])
"""
tdata="sdata/dijet_{0}_{1}/dijet_{0}_{1}_training.root".format(args.pt,int(args.pt*1.1))
vdata="sdata/dijet_{0}_{1}/dijet_{0}_{1}_validation.root".format(args.pt,int(args.pt*1.1))
#print(tdata,vdata)
train=wkiter([tdata,tdata],batch_size=batch_size,end=args.end*1.,istrain=1,rc=rc,onehot=onehot)
valid=wkiter([vdata,vdata],batch_size=batch_size,end=args.end*1.,rc=rc,onehot=onehot)

savename='save/'+str(args.save)
os.system("mkdir "+savename)
os.system("rm "+savename+'/log.log')
plot_model(model,to_file=savename+'/model.png')
print ("train",train.totalnum(),"eval",valid.totalnum())
import logging
logging.basicConfig(filename=savename+'/log.log',level=logging.DEBUG)
logging.info(str(args))
logging.info(str(datetime.datetime.now()))
logging.info(str(train.totalnum())+" batches")
#logger=keras.callbacks.CSVLogger(savename+'/log.log',append=True)
#logger=keras.callbacks.TensorBoard(log_dir=savename+'/logs',histogram_freq=0, write_graph=True , write_images=True, batch_size=20)
history=0
checkpoint=keras.callbacks.ModelCheckpoint(filepath=savename+'/check_{epoch}',monitor='val_loss',verbose=0,save_best_only=False,mode='auto',period=1)

history=model.fit_generator(train.next(),steps_per_epoch=train.totalnum(),validation_data=valid.next(),validation_steps=valid.totalnum(),epochs=epochs,verbose=1,callbacks=[checkpoint])


print(history)
f=open(savename+'/history','w')
try:
  one=history.history['val_acc'].index(max(history.history['val_acc']))
  f.write(str(one)+'\n')
  print(one)
  for i in range(epochs):
    if(i!=one):os.system("rm "+savename+"/check_"+str(i+1))
except:pass
f.write(str(history.history))
f.close()
print (datetime.datetime.now()-start)
logging.info("spent time "+str(datetime.datetime.now()-start))
