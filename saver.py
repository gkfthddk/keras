'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--end",type=float,default=1.,help='end ratio')
parser.add_argument("--save",type=str,default="test_",help='save name')
parser.add_argument("--network",type=str,default="rnn",help='network name on symbols/')
parser.add_argument("--right",type=str,default="/scratch/yjdata/gluon100_img",help='which train sample (qq,gg,zq,zg)')
parser.add_argument("--pt",type=int,default=200,help='pt range pt~pt*1.1')
parser.add_argument("--ptmin",type=float,default=0.,help='pt range pt~pt*1.1')
parser.add_argument("--ptmax",type=float,default=2.,help='pt range pt~pt*1.1')
parser.add_argument("--epochs",type=int,default=10,help='num epochs')
parser.add_argument("--batch_size",type=int,default=512,help='batch_size')
parser.add_argument("--loss",type=str,default="categorical_crossentropy",help='network name on symbols/')
parser.add_argument("--gpu",type=int,default=0,help='gpu number')
parser.add_argument("--isz",type=int,default=0,help='0 or z or not')
parser.add_argument("--channel",type=int,default=64,help='number of sequence channel')
parser.add_argument("--order",type=int,default=1,help='pt ordering')
parser.add_argument("--eta",type=float,default=0.,help='end ratio')
parser.add_argument("--etabin",type=float,default=2.4,help='end ratio')
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
from aaiter import *
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from importlib import import_module
import datetime
start=datetime.datetime.now()
config =tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.3
set_session(tf.Session(config=config))

batch_size = 100000
num_classes = 2 


epochs = args.epochs
print(epochs)

# input image dimensions
img_rows, img_cols = 33, 33

input_shape1 = (10,33,33)
input_shape2 = (20,9)

if(args.loss=="weakloss"):args.loss=weakloss
net=import_module('symbols.symbols')
try:
  onehot=net.onehot(args.network)
  input_shape2=(20,5)
except:onehot=0
model=net.get_symbol(args.network)
rc=""
for sha in model._feed_inputs:
  if(len(sha._keras_shape)==4):
    rc+="c"
  if(len(sha._keras_shape)==3):
    rc+="r"
print("rc",rc)
rc="r"
#model.compile(loss='mean_squared_error',
model.compile(loss=args.loss,
              optimizer=keras.optimizers.Adam(),
	      metrics=['accuracy'])
"""model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])
"""
vzjdata="Data/zj_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
vjjdata="Data/jj_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
vzqdata="Data/zq_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
vzgdata="Data/zg_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
vqqdata="Data/qq_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
vggdata="Data/gg_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
if(args.isz==0):
  tqdata="Data/zj_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  tgdata="Data/jj_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  train=wkiter([tqdata,tgdata],batch_size=batch_size,begin=0,end=args.end*0.7,istrain=1,rc=rc,onehot=onehot,channel=args.channel,order=args.order,eta=args.eta,etabin=args.etabin,pt=args.pt,ptmin=args.ptmin,ptmax=args.ptmax)
  valid1=wkiter([vzjdata,vjjdata],batch_size=batch_size,begin=0.8*args.end,end=args.end*1.,rc=rc,onehot=onehot,channel=args.channel,order=args.order,eta=args.eta,etabin=args.etabin,pt=args.pt,ptmin=args.ptmin,ptmax=args.ptmax)
elif(args.isz==1):
  tqdata="Data/zq_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  tgdata="Data/zg_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  train=wkiter([tqdata,tgdata],batch_size=batch_size,begin=0.6*args.end,end=args.end*1.,istrain=1,rc=rc,onehot=onehot,channel=args.channel,order=args.order)
  valid1=wkiter([vzqdata,vzgdata],batch_size=batch_size,end=args.end*0.2,rc=rc,onehot=onehot,channel=args.channel,order=args.order)
else):
  tqdata="Data/qq_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  tgdata="Data/gg_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  train=wkiter([tqdata,tgdata],batch_size=batch_size,begin=0.6*args.end,end=args.end*1.,istrain=1,rc=rc,onehot=onehot,channel=args.channel,order=args.order)
  valid1=wkiter([vzqdata,vzgdata],batch_size=batch_size,end=args.end*0.2,rc=rc,onehot=onehot,channel=args.channel,order=args.order)
  #valid2=wkiter([vzqdata,vzgdata],batch_size=batch_size,end=2048,rc=rc,onehot=onehot,channel=args.channel,order=args.order,eta=args.eta,etabin=args.etabin,pt=args.pt,ptmin=args.ptmin,ptmax=args.ptmax)
  #valid3=wkiter([vqqdata,vggdata],batch_size=batch_size,end=2048,rc=rc,onehot=onehot,channel=args.channel,order=args.order,eta=args.eta,etabin=args.etabin,pt=args.pt,ptmin=args.ptmin,ptmax=args.ptmax)
print("data ",tqdata)

vx,vy=next(valid1.next())
#logger=keras.callbacks.CSVLogger(savename+'/log.log',append=True)
#logger=keras.callbacks.TensorBoard(log_dir=savename+'/logs',histogram_freq=0, write_graph=True , write_images=True, batch_size=20)
X,Y=next(train.next())
#X=X[0]
np.savez_compressed('aapt{}eta{}isz{}'.format(args.pt,args.etabin,args.isz),X=X,Y=Y,vx=vx,vy=vy)
print("shape",np.array(X).shape)
