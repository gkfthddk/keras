'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--end",type=float,default=100000.,help='end ratio')
parser.add_argument("--save",type=str,default="test_",help='save name')
parser.add_argument("--network",type=str,default="cnn",help='network name on symbols/')
parser.add_argument("--right",type=str,default="/scratch/yjdata/gluon100_img",help='which train sample (qq,gg,zq,zg)')
parser.add_argument("--opt",type=str,default="default",help='optimizer sgd rms adam')
parser.add_argument("--pt",type=int,default=200,help='pt range pt~pt*1.1')
parser.add_argument("--ptmin",type=float,default=0.,help='pt range pt~pt*1.1')
parser.add_argument("--ptmax",type=float,default=2.,help='pt range pt~pt*1.1')
parser.add_argument("--epochs",type=int,default=10,help='num epochs')
parser.add_argument("--batch_size",type=int,default=512,help='batch_size')
parser.add_argument("--loss",type=str,default="categorical_crossentropy",help='network name on symbols/')
parser.add_argument("--gpu",type=int,default=0,help='gpu number')
parser.add_argument("--isz",type=int,default=0,help='0 or z or not')
parser.add_argument("--eta",type=float,default=0.,help='end ratio')
parser.add_argument("--etabin",type=float,default=2.4,help='end ratio')
parser.add_argument("--unscale",type=int,default=1,help='end ratio')
parser.add_argument("--normb",type=float,default=10.,help='end ratio')
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
from miter import *
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from importlib import import_module
import datetime
start=datetime.datetime.now()
config =tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.35
set_session(tf.Session(config=config))

batch_size = args.batch_size
num_classes = 2 


epochs = args.epochs
print(epochs)

net=import_module('symbols.symbols')
if(args.network=="cnn"):
  model=net.modelss()
else:
  model=net.get_symbol(args.network)
rc=""
for sha in model._feed_inputs:
  if(len(sha._keras_shape)==4):
    rc+="c"
  if(len(sha._keras_shape)==3):
    rc+="r"
print("rc",rc)
#model.compile(loss='mean_squared_error',
if(args.opt=="sgd"):
  opt=keras.optimizers.SGD()
if(args.opt=="rms"):
  opt=keras.optimizers.RMSprop()
if(args.opt=="adam"):
  opt=keras.optimizers.Adam()
if(args.opt=="default"):
  if(rc=="c"):
    opt=keras.optimizers.RMSprop()
  if(rc=="r"):
    opt=keras.optimizers.SGD()
model.compile(loss=args.loss,
              optimizer=opt,
	      metrics=['accuracy'])
vjjdata="Data/jj_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))

onehot=0

if(args.isz==0):
  tgdata="Data/jj_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  train=wkiter([tgdata,tgdata],batch_size=batch_size,end=0.7,istrain=1,rc=rc,onehot=onehot,eta=args.eta,etabin=args.etabin,pt=args.pt,ptmin=args.ptmin,ptmax=args.ptmax,unscale=args.unscale,normb=args.normb)
  valid1=wkiter([vjjdata,vjjdata],batch_size=batch_size,begin=0.8,end=1.,rc=rc,onehot=onehot,eta=args.eta,etabin=args.etabin,pt=args.pt,ptmin=args.ptmin,ptmax=args.ptmax,unscale=args.unscale,normb=args.normb)
elif(args.isz==1):
  tqdata="Data/zq_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  tgdata="Data/zg_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  train=wkiter([tqdata,tgdata],batch_size=batch_size,begin=0.6*args.end,end=args.end*1.,istrain=1,rc=rc,onehot=onehot,eta=args.eta,etabin=args.etabin,pt=args.pt,ptmin=args.ptmin,ptmax=args.ptmax,unscale=args.unscale,normb=args.normb)
  #valid1=wkiter([vzjdata,vjjdata],batch_size=batch_size,begin=0.7*args.end,end=args.end*0.7+512,rc=rc,onehot=onehot,eta=args.eta,etabin=args.etabin,pt=args.pt,ptmin=args.ptmin,ptmax=args.ptmax)
  valid1=wkiter([vzqdata,vzgdata],batch_size=batch_size,end=args.end*0.2,rc=rc,onehot=onehot,unscale=args.unscale,normb=args.normb)
  #valid3=wkiter([vqqdata,vggdata],batch_size=batch_size,end=512,rc=rc,onehot=onehot)
else:
  tqdata="Data/qq_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  tgdata="Data/gg_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  train=wkiter([tqdata,tgdata],batch_size=batch_size,begin=0.6*args.end,end=args.end*1.,istrain=1,rc=rc,onehot=onehot,eta=args.eta,etabin=args.etabin,pt=args.pt,ptmin=args.ptmin,ptmax=args.ptmax,unscale=args.unscale,normb=args.normb)
  #valid1=wkiter([vzjdata,vjjdata],batch_size=batch_size,begin=0.7*args.end,end=args.end*0.7+512,rc=rc,onehot=onehot,eta=args.eta,etabin=args.etabin,pt=args.pt,ptmin=args.ptmin,ptmax=args.ptmax)
  #valid2=wkiter([vzqdata,vzgdata],batch_size=batch_size,end=512,rc=rc,onehot=onehot)
  valid1=wkiter([vqqdata,vggdata],batch_size=batch_size,end=args.end*0.2,rc=rc,onehot=onehot,unscale=args.unscale,normb=args.normb)
print("data ",tgdata)

savename='save/'+str(args.save)
#history=AddVal([(valid1,"val1"),(valid2,"val2"),(valid3,"val3")],savename)
history=AddVal([(next(valid1.next()),"val1")],savename)
os.system("mkdir "+savename)
os.system("rm "+savename+'/log.log')
plot_model(model,to_file=savename+'/model.png')
print("### plot done ###")
print ("train",train.totalnum())
import logging
logging.basicConfig(filename=savename+'/log.log',level=logging.DEBUG)
logging.info(str(args))
logging.info(str(datetime.datetime.now()))
logging.info(str(train.totalnum())+" batches")
#logger=keras.callbacks.CSVLogger(savename+'/log.log',append=True)
#logger=keras.callbacks.TensorBoard(log_dir=savename+'/logs',histogram_freq=0, write_graph=True , write_images=True, batch_size=20)
checkpoint=keras.callbacks.ModelCheckpoint(filepath=savename+'/check_{epoch}',monitor='val_loss',verbose=0,save_best_only=False,mode='auto',period=1)
X,Y=next(train.next())
print("shape",np.array(X).shape)
