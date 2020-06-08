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
parser.add_argument("--network",type=str,default="rnn",help='network name on symbols/')
parser.add_argument("--opt",type=str,default="sgd",help='optimizer sgd rms adam')
parser.add_argument("--pt",type=int,default=20,help='pt range pt~pt*1.1')
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
parser.add_argument("--normb",type=float,default=1.,help='end ratio')
parser.add_argument("--stride",type=int,default=1,help='end ratio')
parser.add_argument("--pred",type=int,default=0,help='end ratio')
parser.add_argument("--channel",type=int,default=4,help='end ratio')
parser.add_argument("--mod",type=int,default=0,help='end ratio')
parser.add_argument("--rsect",type=int,default=0,help='rnn section')
parser.add_argument("--dsect",type=int,default=0,help='dense section')
parser.add_argument("--seed",type=str,default="",help='seed of model')
parser.add_argument("--memo",type=str,default="",help='some memo')
args=parser.parse_args()
import os
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Embedding
from keras.layers import Conv2D, MaxPooling2D, SimpleRNN
from keras import backend as K
from numpy.random import seed
seed(101)
#from keras.utils import plot_model
import subprocess
import random
import warnings
import math
from array import array
import numpy as np
#import ROOT as rt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from importlib import import_module
from sklearn.utils import shuffle
import datetime
from sklearn.metrics import roc_auc_score, auc, roc_curve
def valauc(y_true,y_pred):
   #return roc_auc_score(y_true,y_pred)
   print(y_true,y_pred)
   return K.mean(y_pred)
start=datetime.datetime.now()
if(args.gpu!=-1):
  print("gpugpu")
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
  #config =tf.ConfigProto(device_count={'GPU':1})
  #config.gpu_options.per_process_gpu_memory_fraction=0.6
  #set_session(tf.Session(config=config))
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

batch_size = args.batch_size
num_classes = 2 


epochs = args.epochs
print(epochs)

# input image dimensions

if(args.loss=="weakloss"):args.loss=weakloss
net=import_module('symbols.symbols')
if(args.opt=="sgd"):
  opt=keras.optimizers.SGD()
if(args.opt=="rms"):
  opt=keras.optimizers.RMSprop()
if(args.opt=="adam"):
  opt=keras.optimizers.Adam()
losses=args.loss
if(args.stride==2):
  if(args.mod==0):
    losses={"output1" : args.loss,"output2" : args.loss}
    lossweight={"output1" : 1.0, "output2" : 1.0}
  else:
    losses=losses={"output" : args.loss}
    lossweight= {"output" : 1.0}
else:
  losses=losses={"output1" : args.loss}
  lossweight= {"output1" : 1.0}
loaded=np.load("side{}img.npz".format(args.pt))
imgset=loaded["imgset"].item()
rocs=[]
for channels in [[0],[0,1],[0,2],[0,3],[0,1,2],[0,1,3],[0,2,3],[0,1,2,3],[1],[1,2],[1,3],[1,2,3],[2],[2,3],[3]]:
  model=net.drmodel((len(channels),17,17))
  model.compile(loss=losses,
                optimizer=opt, loss_weights=lossweight,
  	      metrics=['accuracy',tf.keras.metrics.AUC()])
  """model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.SGD(),
                metrics=['accuracy'])
  """
  savename='save/'+str(args.save)
  os.system("mkdir -p "+savename)
  os.system("rm "+savename+'/log.log')
  os.system("cp symbols/symbols.py "+savename+'/')
  #plot_model(model,to_file=savename+'/model.png')
  print("### plot done ###")
  import logging
  logging.basicConfig(filename=savename+'/log.log',level=logging.DEBUG)
  logging.info(str(args))
  logging.info(str(datetime.datetime.now()))
  checkpoint=keras.callbacks.ModelCheckpoint(filepath=savename+'/check_{epoch}',monitor='val_loss',verbose=0,save_best_only=False,mode='auto',period=1)
  if("qg" in args.memo):
    el=np.stack([imgset["uj"][:,channel] for channel in channels],axis=1)
    pi=np.stack([imgset["gj"][:,channel] for channel in channels],axis=1)
  
    """el2=[]
    for i in range(len(el)/2):
      el2.append(el[2*i]+el[2*i+1])
    pi2=[]
    for i in range(len(pi)/2):
      pi2.append(pi[2*i]+pi[2*i+1])
    el=np.array(el2)
    pi=np.array(pi2)"""
  if("ep" in args.memo):
    el=np.stack([imgset["el"][:,channel] for channel in channels],axis=1)
    pi=np.stack([imgset["pi"][:,channel] for channel in channels],axis=1)
  ellabel=len(el)*[[1.,0.]]
  pilabel=len(pi)*[[0.,1.]]
  X,Y=shuffle(np.concatenate([el,pi]),np.concatenate([ellabel,pilabel]))
  testX=X[int(0.7*len(X)):]
  testY=Y[int(0.7*len(Y)):]
  X=X[:int(0.7*len(X))]
  Y=Y[:int(0.7*len(Y))]
  print("shape",Y.shape,X.shape)
  if(args.stride==1):history=model.fit(X,Y,batch_size=128,epochs=epochs,verbose=1,validation_split=0.4,callbacks=[checkpoint])
  
  #print(history.history)
  f=open(savename+'/history','w')
  try:
    one=history.history['val_loss'].index(min(history.history['val_loss']))
    f.write(str(one)+'\n')
    print(one)
    for i in range(epochs):
      if(i!=one):os.system("rm "+savename+"/check_"+str(i+1))
  except:
    print("failed to drop")
  f.write(str(history.history))
  f.close()
  print (datetime.datetime.now()-start)
  logging.info("memo "+args.memo)
  logging.info("spent time "+str(datetime.datetime.now()-start))
  logging.info("python jetdualpred.py --save {} --pt {} --stride {} --gpu {} --mod {}".format(args.save,args.pt,args.stride,args.gpu,args.mod))
  import matplotlib.pyplot as plt
  bp=model.predict(testX,verbose=0)
  #bp=model.predict(X[int(0.4*len(X)):],verbose=0)
  fpr,tpr,thresholds=roc_curve(testY[:,0],bp[:,0])
  #fpr,tpr,thresholds=roc_curve(Y[int(0.6*len(Y)):][:,0],bp[:,0])
  fs=25
  tnr=1-fpr
  plt.figure(figsize=(12, 8))
  if("qg" in args.memo):
    plt.xlabel("Quark Efficiency", fontsize=fs*1.2)
    plt.ylabel("Gluon Rejection", fontsize=fs*1.2)
  if("ep" in args.memo):
    plt.xlabel("e- Efficiency", fontsize=fs*1.2)
    plt.ylabel("pi+ Rejection", fontsize=fs*1.2)
  plt.tick_params(labelsize=fs)
  print("AUC:{}".format(round(roc_auc_score(testY[:,0],bp[:,0]),4)))
  label="AUC:{}".format(round(roc_auc_score(testY[:,0],bp[:,0]),4))
  #label="AUC:{}".format(round(roc_auc_score(Y[int(0.6*len(Y)):][:,0],bp[:,0]),4))
  plt.plot(tpr,tnr,lw=3.5,label=label,linestyle="-")
  plt.legend(loc=3, fontsize=fs*0.9)
  plt.grid(alpha=0.6)
  plt.axis((0,1,0,1))
  plt.savefig("{}.png".format(args.memo),bbox_inches='tight',pad_inches=0.5)
  
  rocs.append([channels,roc_auc_score(testY[:,0],bp[:,0])])
print(rocs)
f=open("rocarrange","w")
f.write(str(rocs))
f.close()
if(args.pred==1):
  #os.system("python /home/yulee/keras/jetdualpred.py --save {} --pt {} --stride {} --gpu {} --mod {}".format(args.save,args.pt,args.stride,args.gpu,args.mod))
  savename="save/"+str(args.save)
  history=open(savename+"/history").readlines()
  try:
    try:
      hist=eval(history[0])
      #a=hist['val1_auc']
      a=hist['val_loss']
    except:
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
  
  if(args.stride==1):
    bp=model.predict(X,verbose=0)
    label=Y
  if(args.stride==2):
    bp=model.predict([X[0],X[1]],verbose=0)
    label1=Y[0]
    label2=Y[1]
  if(args.stride==1):
    line1="{} roc {} \n".format(args.save,roc_auc_score(Y[:,0],bp[:,0]))
    print(line1)
    f=open("/hdfs/store/user/yulee/keras/mergelog","a")
    f.write(line1)
  if(args.stride==2):
    line1="{} roc 12 {} {} mean {} ".format(args.save,round(roc_auc_score(label1[:,0],bp[0][:,0]),5),round(roc_auc_score(label2[:,0],bp[1][:,0]),5),round(roc_auc_score(np.concatenate([label1[:,0],label2[:,0]]),np.concatenate([bp[0][:,0],bp[1][:,0]])),5))
    score1=round(roc_auc_score(label1[:,0],bp[0][:,0]),5)
    bp=model.predict([X[0],X[0]],verbose=0)
    line2="{} roc 11 {} {} {} \n".format(args.save,round(roc_auc_score(label1[:,0],bp[0][:,0]),5),round(roc_auc_score(label1[:,0],bp[1][:,0]),5),score1-round(roc_auc_score(label1[:,0],bp[0][:,0]),5))
    print(line1)
    print(line2)
    f.write(line1)
    f.write(line2)
    f.close()
