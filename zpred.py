'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function
import argparse
import keras
from keras import backend as K
from ziter import *
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from importlib import import_module
import matplotlib.pyplot as plt
plt.switch_backend('agg')

batch_size = 512

parser=argparse.ArgumentParser()
parser.add_argument("--end",type=float,default=1.,help='end ratio')
parser.add_argument("--epoch",type=int,default=10,help='epoch')
parser.add_argument("--save",type=str,default="ten100grucnn",help='rch')
parser.add_argument("--rc",type=str,default='rc',help='rnn or cnn')
parser.add_argument("--gpu",type=int,default=0,help='gpu')
parser.add_argument("--pt",type=int,default=100,help='pt range pt~pt*1.1')
parser.add_argument("--isz",type=int,default=0,help='0 or z or not')
parser.add_argument("--channel",type=int,default=30,help='sequence channel')
parser.add_argument("--order",type=int,default=1,help='pt ordering')

args=parser.parse_args()

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
config =tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.3
set_session(tf.Session(config=config))

# input image dimensions
img_rows, img_cols = 33, 33

# the data, shuffled and split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

#if K.image_data_format() == 'channels_first':
#    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#    input_shape = (1, img_rows, img_cols)
#else:
#x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape1= (9,33,33)
input_shape2= (20,10)

#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.Adadelta(),
#              metrics=['accuracy'])
#model=keras.models.load_model('save/fullydijetsame_10')
savename="save/"+str(args.save)
history=open(savename+"/history").readlines()
try:
  hist=eval(history[0])
  a=hist['val1_auc']
except:
  hist=eval(history[1])
vzjdata="Data/zj_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
vjjdata="Data/jj_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
vzqdata="Data/zq_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
vzgdata="Data/zg_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
vqqdata="Data/qq_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
vggdata="Data/gg_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
from sklearn.metrics import roc_auc_score, auc, roc_curve
if(args.isz==0):iii=1
if(args.isz==1):iii=2
if(args.isz==-1):iii=3
epoch=hist['val{}_auc'.format(iii)].index(max(hist['val{}_auc'.format(iii)]))+1
model=keras.models.load_model(savename+"/check_"+str(epoch))
rc=""
for sha in model._feed_inputs:
  if(len(sha._keras_shape)==4):
    rc+="c"
  if(len(sha._keras_shape)==3):
    rc+="r"
onehot=0
test2=wkiter([vzqdata,vzgdata],batch_size=batch_size,begin=args.end*0.2,end=args.end*0.5,rc=rc,onehot=onehot,channel=args.channel,order=args.order)
test3=wkiter([vqqdata,vggdata],batch_size=batch_size,begin=args.end*0.2,end=args.end*0.5,rc=rc,onehot=onehot,channel=args.channel,order=args.order)
entries=test2.totalnum()
print ("test   ",entries)
#epoch=eval(open(savename+"/history").readline())+1
test2.reset()
test3.reset()
for ii in range(2,4):
  if(ii==2):
    if(args.isz==-1):
      continue 
    gen=test2.next()
  if(ii==3):
    if(args.isz==1):
      continue
    gen=test3.next()
  x=[]
  y=[]
  g=[]
  q=[]
  outname=savename+"/v{}t{}".format(iii,ii)
  for j in range(entries):
    a,c=next(gen)
    b=model.predict(a,verbose=0)[:,0]
    x=np.append(x,np.array(c[:,0]))
    y=np.append(y,b)
    for i in range(batch_size):
      if(c[i][0]==0):
        g.append(b[i])
      else:
        q.append(b[i])
  plt.figure(1)
  plt.hist(q,bins=50,weights=np.ones_like(q),histtype='step',alpha=0.7,label='quark')
  plt.hist(g,bins=50,weights=np.ones_like(g),histtype='step',alpha=0.7,label='gluon')
  plt.legend(loc="upper center")
  plt.savefig(outname+"out.png")
  f=open(outname+"out.dat",'w')
  f.write(str(q)+"\n")
  f.write(str(g))
  f.close()
  t_fpr,t_tpr,_=roc_curve(x,y)
  t_tnr=1-t_fpr
  test_auc=np.around(auc(t_fpr,t_tpr),4)
  plt.figure(2)
  plt.plot(t_tpr,t_tnr,alpha=0.5,label="AUC={}".format(test_auc),lw=2)
  plt.legend(loc='lower left')
  os.system("rm "+outname+"roc*.png")
  plt.savefig(outname+"roc"+str(round(test_auc,3))+".png")
  f=open(outname+"roc.dat",'w')
  f.write(str(t_tpr.tolist())+"\n")
  f.write(str(t_tnr.tolist()))
  f.close()
#print(b,c)

