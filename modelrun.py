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
from aiter import *
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from importlib import import_module
import datetime
start=datetime.datetime.now()
config =tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.45
set_session(tf.Session(config=config))

batch_size = args.batch_size
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
model=net.modelss()
rc=""
for sha in model._feed_inputs:
  if(len(sha._keras_shape)==4):
    rc+="c"
  if(len(sha._keras_shape)==3):
    rc+="r"
print("rc",rc)
#model.compile(loss='mean_squared_error',
model.compile(loss=args.loss,
              optimizer=keras.optimizers.RMSprop(),
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
#print(tdata,vdata)
#valid1=wkiter([vzjdata,vjjdata],batch_size=batch_size,begin=0.7*args.end,end=args.end*1.,rc=rc,onehot=onehot)
#valid2=wkiter([vzqdata,vzgdata],batch_size=batch_size,end=args.end*0.3,rc=rc,onehot=onehot)
#valid3=wkiter([vqqdata,vggdata],batch_size=batch_size,end=args.end*0.3,rc=rc,onehot=onehot)
if(args.isz==0):
  tqdata="Data/zj_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  tgdata="Data/jj_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  train=wkiter([tqdata,tgdata],batch_size=batch_size,end=args.end*0.7,istrain=1,rc=rc,onehot=onehot,eta=args.eta,etabin=args.etabin,pt=args.pt,ptmin=args.ptmin,ptmax=args.ptmax)
  valid1=wkiter([vzjdata,vjjdata],batch_size=batch_size,begin=0.8*args.end,end=args.end*1.,rc=rc,onehot=onehot,eta=args.eta,etabin=args.etabin,pt=args.pt,ptmin=args.ptmin,ptmax=args.ptmax)
  #valid2=wkiter([vzqdata,vzgdata],batch_size=batch_size,end=2048,rc=rc,onehot=onehot)
  #valid3=wkiter([vqqdata,vggdata],batch_size=batch_size,end=2048,rc=rc,onehot=onehot)
elif(args.isz==1):
  tqdata="Data/zq_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  tgdata="Data/zg_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  train=wkiter([tqdata,tgdata],batch_size=batch_size,begin=0.5*args.end,end=args.end*1.,istrain=1,rc=rc,onehot=onehot,eta=args.eta,etabin=args.etabin)
  valid1=wkiter([vzjdata,vjjdata],batch_size=batch_size,begin=0.7*args.end,end=args.end*0.7+512,rc=rc,onehot=onehot,eta=args.eta,etabin=args.etabin)
  #valid2=wkiter([vzqdata,vzgdata],batch_size=batch_size,end=args.end*0.1,rc=rc,onehot=onehot)
  #valid3=wkiter([vqqdata,vggdata],batch_size=batch_size,end=512,rc=rc,onehot=onehot)
else:
  tqdata="Data/qq_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  tgdata="Data/gg_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  train=wkiter([tqdata,tgdata],batch_size=batch_size,begin=0.5*args.end,end=args.end*1.,istrain=1,rc=rc,onehot=onehot,eta=args.eta,etabin=args.etabin)
  valid1=wkiter([vzjdata,vjjdata],batch_size=batch_size,begin=0.7*args.end,end=args.end*0.7+512,rc=rc,onehot=onehot,eta=args.eta,etabin=args.etabin)
  #valid2=wkiter([vzqdata,vzgdata],batch_size=batch_size,end=512,rc=rc,onehot=onehot)
  #valid3=wkiter([vqqdata,vggdata],batch_size=batch_size,end=args.end*0.1,rc=rc,onehot=onehot)
print("data ",tqdata)

savename='save/'+str(args.save)
#history=AddVal([(valid1,"val1"),(valid2,"val2"),(valid3,"val3")],savename)
history=AddVal([(valid1,"val1")],savename)
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
model.fit_generator(train.next(),steps_per_epoch=train.totalnum(),epochs=epochs,verbose=1,callbacks=[checkpoint,history])

print(history)
f=open(savename+'/history','w')
try:
  one=history.history['val1_auc'].index(max(history.history['val1_auc']))
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
