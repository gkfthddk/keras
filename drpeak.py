#!/usr/bin/python3
#!/usr/bin/python2.7
'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
#python jetdual.py --save dualn2200 --network nnn2 --pt 200 --epoch 50 --stride 2 --gpu 3
#python jetdual.py --save dualn2m2200 --network nnn2 --pt 200 --epoch 50 --stride 2 --gpu 4 --pred 1 --mod 2
from __future__ import print_function
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--end",type=float,default=1.,help='end ratio')
parser.add_argument("--save",type=str,default="test",help='save name')
parser.add_argument("--network",type=str,default="dr3dmodel",help='network name on symbols/')
parser.add_argument("--opt",type=str,default="sgd",help='optimizer sgd rms adam')
parser.add_argument("--pt",type=int,default=20,help='pt range pt~pt*1.1')
parser.add_argument("--ptmin",type=float,default=0.,help='pt range pt~pt*1.1')
parser.add_argument("--ptmax",type=float,default=2.,help='pt range pt~pt*1.1')
parser.add_argument("--epochs",type=int,default=10,help='num epochs')
parser.add_argument("--batch_size",type=int,default=512,help='batch_size')
parser.add_argument("--loss",type=str,default="categorical_crossentropy",help='network name on symbols/')
parser.add_argument("--gpu",type=int,default=0,help='gpu number')
parser.add_argument("--voxel",type=int,default=0,help='0 or z or not')
parser.add_argument("--target",type=int,default=1,help='end ratio')
parser.add_argument("--pix",type=int,default=90,help='end ratio')
parser.add_argument("--num_files",type=int,default=500,help='end ratio')
parser.add_argument("--stride",type=str,default="1,2,3,4",help='end ratio')
parser.add_argument("--num_point",type=int,default=2048,help='end ratio')
parser.add_argument("--channel",type=int,default=4,help='end ratio')
parser.add_argument("--dform",type=str,default="pixel",help='pixel voxel point')
parser.add_argument("--mod",type=int,default=0,help='end ratio')
parser.add_argument("--rot",type=int,default=1,help='pixel num')
parser.add_argument("--seed",type=str,default="",help='seed of model')
parser.add_argument("--memo",type=str,default="",help='some memo')
args=parser.parse_args()
import os,sys
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
#import keras as keras
#from keras.models import Sequential, Model
#from keras.layers import *
from numpy.random import seed
#seed(101)
import subprocess
import random
import warnings
import math
import shutil
from array import array
import numpy as np
#import ROOT as rt
import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
from importlib import import_module
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
import datetime
from sklearn.metrics import roc_auc_score, auc, roc_curve
from driter import *
start=datetime.datetime.now()
if(args.gpu!=-1):
  print("gpugpu")
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
  #config =tf.ConfigProto(device_count={'GPU':1})
  #config.gpu_options.per_process_gpu_memory_fraction=0.6
  #set_session(tf.Session(config=config))
  #gpus = tf.config.experimental.list_physical_devices('GPU')
  #tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

batch_size = args.batch_size
num_classes = 2 


epochs = args.epochs
print(epochs)
# input image dimensions
net=import_module('symbols.symbols')
channel=args.channel
if(args.opt=="sgd"):
  opt=keras.optimizers.SGD()
if(args.opt=="rms"):
  opt=keras.optimizers.RMSprop()
if(args.opt=="adam"):
  opt=keras.optimizers.Adam()
losses=args.loss
losses={}
lossweight={}
stride=[int(i) for i in args.stride.split(",")]
for i in stride:
  losses["output{}".format(i)]=args.loss
  lossweight["output{}".format(i)]=1.0
  #losses={"output1" : args.loss}
  #lossweight= {"output1" : 1.0}

data_path=["/pad/yulee/geant4/tester/analysis/fast/uJet50GeV_fastsim_{}.root","/pad/yulee/geant4/tester/analysis/fast/gJet50GeV_fastsim_{}.root"]
traindata,valdata,testdata=prepare_data(data_path,args.num_files,data_form=args.dform,batch_size=batch_size,num_channel=channel,num_point=args.num_point,pix=args.pix,target=1,stride=stride)
if(args.dform=="pixel"):
    model=net.peakmodelc(traindata.data_shape,stride)
if(args.dform=="point"):
  model=pointmodel(args.num_point,channel,peak=1)
if(args.dform=="voxel"):
  model=net.dr3dmodel3(traindata.data_shape)
if(sys.version_info[0]>=3):
  model_metrics=['accuracy',keras.metrics.AUC()]
else:
  model_metrics=['accuracy']
model.compile(loss=losses,
              optimizer=opt, loss_weights=lossweight,
        metrics=["mse","mae"])


savename='save/'+str(args.save)
if(args.memo==""):args.memo=args.save
if(not os.path.isdir(savename)):
  os.makedirs(savename)
if os.path.isfile(savename+'/log.log'):
  os.remove(savename+'/log.log')
shutil.copy("symbols/symbols.py",savename+'/')
shutil.copy(__file__,savename+'/')
#from keras.utils import plot_model
#plot_model(model,to_file=savename+'/model.png')
#plot_model(model,to_file='/home/yulee/keras/model.png')
print("### plot done ###")
import logging
logging.basicConfig(filename=savename+'/log.log',level=logging.DEBUG)
logging.info(str(args))
logging.info(str(datetime.datetime.now()))
checkpoint=keras.callbacks.ModelCheckpoint(filepath=savename+'/check_{epoch}',monitor='val_loss',verbose=0,save_weights_only=False,save_best_only=False,mode='auto',save_freq="epoch")
print(model.summary())
print("shape",traindata.total_len,traindata.data_shape)
history=model.fit(traindata,epochs=epochs,validation_data=valdata,verbose=1,callbacks=[checkpoint])

#print(history.history)
f=open(savename+'/history','w')
one=history.history['val_loss'].index(min(history.history['val_loss']))
f.write(str(one)+'\n')
print(one)
for i in range(epochs):
  try:
    if(i!=one):shutil.rmtree(savename+"/check_"+str(i+1),ignore_errors=True)
  except:
    print("failed to drop")
model=keras.models.load_model(savename+"/check_"+str(one+1))
print("epoch {} loaded".format(one+1))

f.write(str(history.history))
f.close()
print (datetime.datetime.now()-start)
logging.info("memo "+args.memo)
logging.info("spent time "+str(datetime.datetime.now()-start))
logging.info("python jetdualpred.py --save {} --pt {} --stride {} --gpu {} --mod {}".format(args.save,args.pt,stride,args.gpu,args.mod))
import matplotlib.pyplot as plt
testX,testY=testdata.get_test()
bp=model.predict(testX,verbose=0)
#bp=model.predict(X[int(0.4*len(X)):],verbose=0)
from sklearn.metrics import mean_squared_error, mean_absolute_error
y=[]
name=[]
for i in range(len(stride)):
  mseloss=mean_squared_error(testY["output{}".format(stride[i])],bp[i])
  maeloss=mean_absolute_error(testY["output{}".format(stride[i])],bp[i])
  y.append(testY["output{}".format(stride[i])])
  name.append("output{}".format(stride[i]))
  print("ourput {} test mse".format(stride[i]),mseloss,"test mae",maeloss)
#fpr,tpr,thresholds=roc_curve(Y[int(0.6*len(Y)):][:,0],bp[:,0])
np.savez("/pad/yulee/keras/drbox/{}out".format(args.memo),y=y,p=bp,name=name)
#label="AUC:{}".format(round(roc_auc_score(Y[int(0.6*len(Y)):][:,0],bp[:,0]),4))
