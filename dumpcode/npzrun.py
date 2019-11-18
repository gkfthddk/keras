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
parser.add_argument("--unscale",type=int,default=0,help='end ratio')
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
config.gpu_options.per_process_gpu_memory_fraction=0.35
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
model.compile(loss=args.loss,
              optimizer=opt,
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

savename='save/'+str(args.save)
#history=AddVal([(valid1,"val1"),(valid2,"val2"),(valid3,"val3")],savename)
os.system("mkdir "+savename)
os.system("rm "+savename+'/log.log')
plot_model(model,to_file=savename+'/model.png')
print("### plot done ###")
import logging
logging.basicConfig(filename=savename+'/log.log',level=logging.DEBUG)
logging.info(str(args))
logging.info(str(datetime.datetime.now()))
#logger=keras.callbacks.CSVLogger(savename+'/log.log',append=True)
#logger=keras.callbacks.TensorBoard(log_dir=savename+'/logs',histogram_freq=0, write_graph=True , write_images=True, batch_size=20)
checkpoint=keras.callbacks.ModelCheckpoint(filepath=savename+'/check_{epoch}',monitor='val_loss',verbose=0,save_best_only=False,mode='auto',period=1)
if(args.isz==1):
  if(args.etabin==1):
    loaded=np.load("newzqmixed{}pteta.npz".format(args.pt))
    print("newzqmixed{}pteta.npz".format(args.pt))
  else:
    loaded=np.load("newzqmixed{}pt.npz".format(args.pt))
    print("newzqmixed{}pt.npz".format(args.pt))
elif(args.isz==-1):
  if(args.etabin==1):
    loaded=np.load("newqqmixed{}pteta.npz".format(args.pt))
    print("newqqmixed{}pteta.npz".format(args.pt))
  else:
    loaded=np.load("newqqmixed{}pt.npz".format(args.pt))
    print("newqqmixed{}pt.npz".format(args.pt))
elif(args.isz==0):
  if(args.etabin==1):
    if(args.unscale==1):
      loaded=np.load("newunscalemixed{}pteta.npz".format(args.pt))
    else:
      loaded=np.load("newjjmixed{}pteta.npz".format(args.pt))
    print("etabin 1")
  else:
    if(args.unscale==1):
      loaded=np.load("newunscalemixed{}pt.npz".format(args.pt))
    else:
      loaded=np.load("newzjmixed{}pt.npz".format(args.pt))
    print("etabin 2.4")
if(rc=="r"):
  data=loaded["rnnset"]
elif(rc=="c"):
  data=loaded["cnnset"]
label=loaded["label"]
line=int(45000)
endline=int(60000)
if(len(label)<60000):
  line=int(len(label)*3./4.)
  endline=len(label)
X=data[0:line]
vx=data[line:endline]
Y=label[0:line]
vy=label[line:endline]
print(len(X),len(vx),len(Y),len(vy))
history=AddVal([[[vx,vy],"val1"]],savename)
model.fit(X,Y,batch_size=512,epochs=epochs,verbose=1,callbacks=[checkpoint,history])
#model.fit_generator(train.next(),steps_per_epoch=train.totalnum(),epochs=epochs,verbose=1,callbacks=[checkpoint,history])

print(history)
f=open(savename+'/history','w')
try:
  one=history.history['val1_loss'].index(min(history.history['val1_loss']))
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
