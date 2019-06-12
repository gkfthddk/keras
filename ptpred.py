'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function
import argparse
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras
from keras import backend as K
from aiter import *
from keras.backend.tensorflow_backend import set_session
from importlib import import_module
import matplotlib.pyplot as plt
plt.switch_backend('agg')

batch_size = 100000

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
parser.add_argument("--eta",type=float,default=0.,help='end ratio')
parser.add_argument("--etabin",type=float,default=2.4,help='end ratio')

args=parser.parse_args()

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
test2=wkiter([vzqdata,vzgdata],batch_size=batch_size,begin=args.end*0.2,end=args.end*0.6,rc=rc,onehot=onehot,channel=args.channel,order=args.order,eta=args.eta,etabin=args.etabin,pt=args.pt)
test3=wkiter([vqqdata,vggdata],batch_size=batch_size,begin=args.end*0.2,end=args.end*0.6,rc=rc,onehot=onehot,channel=args.channel,order=args.order,eta=args.eta,etabin=args.etabin,pt=args.pt)
entries=test2.totalnum()
print ("test   ",entries)
#epoch=eval(open(savename+"/history").readline())+1
test2.reset()
test3.reset()
for ii in range(2,4):
  if(ii==2):
    if(args.isz==-1):
      continue 
    gen=test2
  if(ii==3):
    if(args.isz==1):
      continue
    gen=test3
  x=[]
  y=[]
  g=[]
  q=[]
  outname=savename+"/v{}t{}".format(iii,ii)
  for j in range(1):
    p=model.predict(gen.gjetset,verbose=0)[:,0]
    pt=gen.gptset
    eta=gen.getaset
    f=open(outname+"ptg.dat",'w')
    f.write(str(p.tolist())+"\n")
    f.write(str(pt.tolist())+"\n")
    f.write(str(eta.tolist()))
    f.close()
    p=model.predict(gen.qjetset,verbose=0)[:,0]
    pt=gen.qptset
    eta=gen.qetaset
    f=open(outname+"ptq.dat",'w')
    f.write(str(p.tolist())+"\n")
    f.write(str(pt.tolist())+"\n")
    f.write(str(eta.tolist()))
    f.close()
    print("###number",args.pt,args.eta)
