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
parser.add_argument("--normb",type=float,default=1.,help='end ratio')
parser.add_argument("--stride",type=int,default=2,help='end ratio')
parser.add_argument("--pred",type=int,default=0,help='end ratio')
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
img_rows, img_cols = 33, 33

if(args.loss=="weakloss"):args.loss=weakloss
net=import_module('/home/yulee/keras/symbols.symbols')
try:
  onehot=net.onehot(args.network)
except:onehot=0
if(args.mod==0):
  #if(args.network=="cnn"):model=net.jetcv(args.stride,args.seed)
  #if(args.network=="cnn"):model=net.modelss((10,33,33))
  if(args.network=="cnn"):model=net.jetcnn(args.stride,args.seed)
  else:model=net.jetcon(args.network,args.stride,args.seed,args.rsect,args.dsect)
else:model=net.jetconmod(args.network,args.stride,args.mod)
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
model.compile(loss=losses,
              optimizer=opt, loss_weights=lossweight,
	      metrics=['accuracy'])
"""model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])
"""
savename='/home/yulee/keras/save/'+str(args.save)
os.system("mkdir "+savename)
os.system("rm "+savename+'/log.log')
os.system("cp symbols/symbols.py "+savename+'/')
#plot_model(model,to_file=savename+'/model.png')
print("### plot done ###")
import logging
logging.basicConfig(filename=savename+'/log.log',level=logging.DEBUG)
logging.info(str(args))
logging.info(str(datetime.datetime.now()))
checkpoint=keras.callbacks.ModelCheckpoint(filepath=savename+'/check_{epoch}',monitor='val_loss',verbose=0,save_best_only=False,mode='auto',period=1)
if(args.network=="cnn"):
  loaded=np.load("/home/yulee/keras/gennc{}.npz".format(args.pt))
  X=loaded["imgset"]
else:
  loaded=np.load("/home/yulee/keras/gendr128{}.npz".format(args.pt))
  X=loaded["seqset"][:,:,:,:]
Y=loaded["labelset"]

Xv=X[:2,int(90000*0.7):90000]
Yv=Y[:2,int(90000*0.7):90000]
Xv=np.concatenate([Xv,[Xv[1],Xv[0]]],axis=1)
Yv=np.concatenate([Yv,[Yv[1],Yv[0]]],axis=1)
Xv[0],Xv[1],Yv[0],Yv[1]=shuffle(Xv[0],Xv[1],Yv[0],Yv[1])

X=X[:2,:int(90000*0.7)]
Y=Y[:2,:int(90000*0.7)]
X=np.concatenate([X,[X[1],X[0]]],axis=1)
Y=np.concatenate([Y,[Y[1],Y[0]]],axis=1)
X[0],X[1],Y[0],Y[1]=shuffle(X[0],X[1],Y[0],Y[1])

X=np.concatenate([X,Xv],axis=1)
Y=np.concatenate([Y,Yv],axis=1)
pidset=loaded["pidset"]
if(args.stride==1):
  #X=X.reshape((-1,10,33,33))
  #Y=Y.reshape((-1,2))
  X=X[0]
  Y=Y[0]
print("shape",Y.shape,X.shape)
if(args.stride==1):history=model.fit(X,Y,batch_size=512,epochs=epochs,verbose=1,validation_split=0.3,callbacks=[checkpoint])
if(args.stride==2):
  if(args.mod==0):
    history=model.fit([X[0],X[1]],{"output1" : Y[0],"output2" : Y[1]},batch_size=512,epochs=epochs,verbose=1,validation_split=0.3,callbacks=[checkpoint])
  else:
    history=model.fit(X,Y,batch_size=512,epochs=epochs,verbose=1,validation_split=0.3,callbacks=[checkpoint])

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

if(args.pred==1):
  #os.system("python /home/yulee/keras/jetdualpred.py --save {} --pt {} --stride {} --gpu {} --mod {}".format(args.save,args.pt,args.stride,args.gpu,args.mod))
  savename="/home/yulee/keras/save/"+str(args.save)
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
  if("c" in rc):
    X=loaded["imgset"]
  else:
    X=loaded["seqset"][:2,:,:,:]
    len(np.unique(X[:,:,:,4]))
  Y=loaded["labelset"]
  X=X[:2,90000:122000]
  Y=Y[:2,90000:122000]
  f=rt.TFile("{}/get.root".format(savename),"recreate")
  qs=[]
  gs=[]
  p=array('f',[0.])
  pt=array('f',[0.])
  eta=array('f',[0.])
  pid=array('f',[0.])
  ptd=array('f',[0.])
  axis1=array('f',[0.])
  axis2=array('f',[0.])
  cmult=array('f',[0.])
  nmult=array('f',[0.])
  trees={}
  for i in range(2):
    for jetid in ["q","g"]:
      trees["{}{}".format(jetid,i)]=rt.TTree("{}{}".format(jetid,i+1),"{}{} tree".format(jetid,i+1))
      trees["{}{}".format(jetid,i)].Branch("p",p,"p/F")
      trees["{}{}".format(jetid,i)].Branch("pt",pt,"pt/F")
      trees["{}{}".format(jetid,i)].Branch("eta",eta,"eta/F")
      trees["{}{}".format(jetid,i)].Branch("phi",eta,"phi/F")
      trees["{}{}".format(jetid,i)].Branch("pid",pid,"pid/F")
      trees["{}{}".format(jetid,i)].Branch("ptd",ptd,"ptd/F")
      trees["{}{}".format(jetid,i)].Branch("cmult",cmult,"cmult/F")
      trees["{}{}".format(jetid,i)].Branch("nmult",nmult,"nmult/F")
      trees["{}{}".format(jetid,i)].Branch("axis1",axis1,"axis1/F")
      trees["{}{}".format(jetid,i)].Branch("axis2",axis2,"axis2/F")
  label1=Y[0]
  label2=Y[1]
  if(args.stride==1):
    #X=X.reshape((-1,10,33*33))
    Y=Y.reshape((-1,2))
  x=[]
  y=[]
  g=[]
  q=[]
  if(args.stride==1):bp=model.predict(X,verbose=0)
  if(args.stride==2):bp=model.predict([X[0],X[1]],verbose=0)
  bpt=loaded["ptset"][:2,90000:122000]
  beta=loaded["etaset"][:2,90000:122000]
  bpid=loaded["pidset"][:2,90000:122000]
  bbdt=loaded["bdtset"][90000:122000]
  chek=[]
  if(args.stride==2):
    if(args.mod==0):leng= len(bp[0])
    else:leng=len(bp)
    for i in range(leng):
      for j in range(args.stride):
        if(label1[i][j]==1):
          if(args.mod==0):p[0]=bp[0][i][j]
          else:p[0]=bp[i][2*j]+bp[i][2*j+1]
          pt[0]=bpt[0][i]
          eta[0]=beta[0][i]
          pid[0]=bpid[0][i]
          ptd[0]=bbdt[i][2]
          cmult[0]=bbdt[i][0]
          nmult[0]=bbdt[i][1]
          axis1[0]=bbdt[i][3]
          axis2[0]=bbdt[i][4]
          trees["{}{}".format(["q","g"][j],0)].Fill()
        if(label2[i][j]==1):
          if(args.mod==0):p[0]=bp[1][i][j]
          else:p[0]=bp[i][j]+bp[i][2+j]
          pt[0]=bpt[1][i]
          eta[0]=beta[1][i]
          pid[0]=bpid[1][i]
          ptd[0]=bbdt[i][7]
          cmult[0]=bbdt[i][5]
          nmult[0]=bbdt[i][6]
          axis1[0]=bbdt[i][8]
          axis2[0]=bbdt[i][9]
          trees["{}{}".format(["q","g"][j],1)].Fill()
  f.Write()
  f.Close()
  if(args.stride==1):
    line1="{} roc {} ".format(args.save,round(roc_auc_score(Y[:,0],bp[:,0]),5))
    print(line1)
    f=open("/home/yulee/keras/mergelog","a")
    f.write(line1)
  if(args.stride==2):
    line1="{} roc 12 {} {} mean {} ".format(args.save,round(roc_auc_score(label1[:,0],bp[0][:,0]),5),round(roc_auc_score(label2[:,0],bp[1][:,0]),5),round(roc_auc_score(np.concatenate([label1[:,0],label2[:,0]]),np.concatenate([bp[0][:,0],bp[1][:,0]])),5))
    score1=round(roc_auc_score(label1[:,0],bp[0][:,0]),5)
    bp=model.predict([X[0],X[0]],verbose=0)
    line2="{} roc 11 {} {} {} \n".format(args.save,round(roc_auc_score(label1[:,0],bp[0][:,0]),5),round(roc_auc_score(label1[:,0],bp[1][:,0]),5),score1-round(roc_auc_score(label1[:,0],bp[0][:,0]),5))
    print(line1)
    print(line2)
    f=open("/home/yulee/keras/mergelog","a")
    f.write(line1)
    f.write(line2)
    f.close()
