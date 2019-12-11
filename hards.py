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
parser.add_argument("--pt",type=int,default=200,help='pt range pt~pt*1.1')
parser.add_argument("--ptmin",type=float,default=0.,help='pt range pt~pt*1.1')
parser.add_argument("--ptmax",type=float,default=2.,help='pt range pt~pt*1.1')
parser.add_argument("--gpu",type=int,default=0,help='gpu number')
parser.add_argument("--isz",type=int,default=0,help='0 or z or not')
parser.add_argument("--eta",type=float,default=0.,help='end ratio')
parser.add_argument("--etabin",type=float,default=2.4,help='end ratio')
parser.add_argument("--unscale",type=int,default=1,help='end ratio')
parser.add_argument("--normb",type=float,default=1.,help='end ratio')
parser.add_argument("--stride",type=int,default=2,help='end ratio')
parser.add_argument("--pred",type=int,default=0,help='end ratio')
parser.add_argument("--mod",type=int,default=0,help='end ratio')
args=parser.parse_args()
import os
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Embedding
from keras.layers import Conv2D, MaxPooling2D, SimpleRNN
from keras import backend as K
#from keras.utils import plot_model
import subprocess
import random
import warnings
import math
from array import array
import numpy as np
import ROOT as rt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from importlib import import_module
from sklearn.utils import shuffle
import datetime
start=datetime.datetime.now()
if(args.gpu!=-1):
  print("gpugpu")
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
  config =tf.ConfigProto(device_count={'GPU':1})
  config.gpu_options.per_process_gpu_memory_fraction=0.6
  set_session(tf.Session(config=config))

num_classes = 2 


loaded=np.load("/home/yulee/keras/gendr128{}.npz".format(args.pt))

# input image dimensions
#os.system("python /home/yulee/keras/jetdualpred.py --save {} --pt {} --stride {} --gpu {} --mod {}".format(args.save,args.pt,args.stride,args.gpu,args.mod))
savename="/home/yulee/keras/save/"+str(args.save)
history=open(savename+"/history").readlines()
try:
  hist=eval(history[0])
  #a=hist['val1_auc']
  a=hist['val_loss']
except:
  sepoch=eval(history[0])
  hist=eval(history[1])
from sklearn.metrics import roc_auc_score, auc, roc_curve
if(args.isz==0):iii=1
if(args.isz==1):iii=2
if(args.isz==-1):iii=3
epoch=hist['val_loss'.format(iii)].index(min(hist['val_loss'.format(iii)]))+1
try:
  epoch=hist['val_loss'.format(iii)].index(min(hist['val_loss'.format(iii)]))+1
  model=keras.models.load_model(savename+"/check_"+str(epoch))
except:
  epoch=sepoch+1
  model=keras.models.load_model(savename+"/check_"+str(epoch))
rc=""
for sha in model._feed_inputs:
  if(sha._keras_shape[-1]==33*33):
    rc+="c"
  if(sha._keras_shape[-1]==33):
    rc+="c"
onehot=0
X=loaded["seqset"][:2,:,:,:]
len(np.unique(X[:,:,:,4]))
Y=loaded["labelset"]
X=X[:2]
Y=Y[:2]
beta=loaded["etaset"]
bphi=loaded["phiset"]
bpt=loaded["ptset"]
bbdt=loaded["bdtset"]
bpid=loaded["pidset"]
beve=loaded["eveset"]

if(args.stride==1):
  #X=X.reshape((-1,10,33*33))
  Y=Y.reshape((-1,2))
x=[]
y=[]
g=[]
q=[]
if(args.stride==1):bp=model.predict(X,verbose=0)
if(args.stride==2):bp=model.predict([X[0],X[1]],verbose=0)
hardx=[]
hardy=[]
hardeta=[]
hardphi=[]
hardpt=[]
hardbdt=[]
hardpid=[]
hardeve=[]
for i in range(len(X[0])):
  for k in range(2):
    if(Y[k][i][0]==1 and bp[k][i][0]<0.5):
      hardx.append(X[k][i])
      hardy.append(Y[k][i])
      hardeta.append(beta[k][i])
      hardphi.append(bphi[k][i])
      hardpt.append(bpt[k][i])
      hardpid.append(bpid[k][i])
      hardeve.append(beve[i])
      hardbdt.append(bbdt[i])
    if(Y[k][i][0]==0 and bp[k][i][0]>0.5):
      hardx.append(X[k][i])
      hardy.append(Y[k][i])
      hardeta.append(beta[k][i])
      hardphi.append(bphi[k][i])
      hardpt.append(bpt[k][i])
      hardpid.append(bpid[k][i])
      hardeve.append(beve[i])
      hardbdt.append(bbdt[i])
hardx=np.array(hardx)
hardy=np.array(hardy)
hardeta=np.array(hardeta)
hardphi=np.array(hardphi)
hardpt=np.array(hardpt)
hardeve=np.array(hardeve)

np.savez_compressed("/home/yulee/keras/hardgen{}".format(args.pt),ptset=hardpt,etaset=hardeta,phiset=hardphi,pidset=hardpid,seqset=hardx,labelset=hardy,eveset=hardeve,bdtset=hardbdt)
print(len(hardeve))
