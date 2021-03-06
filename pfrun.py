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
parser.add_argument("--seed",type=str,default="",help='seed of model')
args=parser.parse_args()
import os
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Embedding
from keras.layers import Conv2D, MaxPooling2D, SimpleRNN
from keras import backend as K
from keras.utils import plot_model
from jetiter import *
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from importlib import import_module
from sklearn.utils import shuffle
import datetime
start=datetime.datetime.now()
if(args.gpu!=-1):
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
  config =tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction=0.3
  set_session(tf.Session(config=config))

batch_size = args.batch_size
num_classes = 2 


epochs = args.epochs
print(epochs)

# input image dimensions

if(args.loss=="weakloss"):args.loss=weakloss
net=import_module('symbols.symbols')
try:
  onehot=net.onehot(args.network)
except:onehot=0
model=net.pfcon(args.network,args.stride,args.seed)
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
plot_model(model,to_file=savename+'/model.png')
print("### plot done ###")
import logging
logging.basicConfig(filename=savename+'/log.log',level=logging.DEBUG)
logging.info(str(args))
logging.info(str(datetime.datetime.now()))
checkpoint=keras.callbacks.ModelCheckpoint(filepath=savename+'/check_{epoch}',monitor='val_loss',verbose=0,save_best_only=False,mode='auto',period=1)
loaded=np.load("/home/yulee/keras/pf{}.npz".format(args.pt))
if("c" in args.network):
  X=loaded["imgset"]
else:
  X=loaded["seqset"][:,:,:4]
Y=loaded["labelset"]

X=X[:int(80000)]
Y=Y[:int(80000)]

pidset=loaded["pidset"]

print("shape",Y.shape,X.shape)
if(args.stride==1):history=model.fit(X,Y,batch_size=512,epochs=epochs,verbose=1,validation_split=0.3,callbacks=[checkpoint])
if(args.stride==2):
  if(args.mod==0):
    history=model.fit(X,{"output1" : Y[:,0],"output2" : Y[:,1]},batch_size=512,epochs=epochs,verbose=1,validation_split=0.3,callbacks=[checkpoint])
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
logging.info("spent time "+str(datetime.datetime.now()-start))
logging.info("python pfpred.py --save {} --pt {} --stride {} --gpu {} --mod {}".format(args.save,args.pt,args.stride,args.gpu,args.mod))

if(args.pred==1):os.system("python /home/yulee/keras/pfpred.py --save {} --pt {} --stride {} --gpu {} --mod {}".format(args.save,args.pt,args.stride,args.gpu,args.mod))
#python jetdualpred.py --save dualn2500 --pt 500 --stride 2 --gpu 3
