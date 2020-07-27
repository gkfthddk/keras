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
parser.add_argument("--voxel",type=int,default=0,help='0 or z or not')
parser.add_argument("--eta",type=float,default=0.,help='end ratio')
parser.add_argument("--etabin",type=float,default=2.4,help='end ratio')
parser.add_argument("--order",type=int,default=1,help='end ratio')
parser.add_argument("--rot",type=int,default=0,help='end ratio')
parser.add_argument("--stride",type=int,default=1,help='end ratio')
parser.add_argument("--pred",type=int,default=0,help='end ratio')
parser.add_argument("--channel",type=int,default=4,help='end ratio')
parser.add_argument("--mod",type=int,default=0,help='end ratio')
parser.add_argument("--num_pnt",type=int,default=1024,help='rnn section')
parser.add_argument("--seed",type=str,default="",help='seed of model')
parser.add_argument("--memo",type=str,default="",help='some memo')
args=parser.parse_args()
import os,sys,shutil
#import tensorflow.keras as keras
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import *
import keras as keras
from keras.models import Model
from keras.layers import *
from numpy.random import seed
#seed(101)
#from keras.utils import plot_model
import subprocess
import random
import warnings
import math
from array import array
import numpy as np
#import ROOT as rt
import tensorflow as tf
from importlib import import_module
from sklearn.utils import shuffle
import datetime
from sklearn.metrics import roc_auc_score, auc, roc_curve
start=datetime.datetime.now()
if(args.gpu!=-1):
  print("gpugpu")
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
  #config =tf.ConfigProto(device_count={'GPU':1})
  #config.gpu_options.per_process_gpu_memory_fraction=0.6
  #set_session(tf.Session(config=config))
  #gpus = tf.config.experimental.list_physical_devices('GPU')
  #tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

batch_size = args.batch_size
num_classes = 2 


epochs = args.epochs
print(epochs)

# input image dimensions
if(args.loss=="weakloss"):args.loss=weakloss
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
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 15:39:40 2017

@author: Gary
"""

def mat_mul(A, B):
    return tf.linalg.matmul(A, B)


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[1, 0, 0, 0],
                                    [0,cosval, -sinval, 0],
                                    [0, sinval, cosval, 0],
                                    [0, 0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 4)),rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


# number of points in each sample
if("gen" in args.memo):
  num_points = 512
else:
  num_points = args.num_pnt 

# number of categories
k = 2

channel= 4

# define optimizer
adam = keras.optimizers.Adam(lr=0.001, decay=0.7)

class MatMul(Layer):

    def __init__(self, **kwargs):
        super(MatMul, self).__init__(**kwargs)

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list):
            raise ValueError('`MatMul` layer should be called '
                             'on a list of inputs')
        if len(input_shape) != 2:
            raise ValueError('The input of `MatMul` layer should be a list containing 2 elements')

        if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
            raise ValueError('The dimensions of each element of inputs should be 3')

        if input_shape[0][-1] != input_shape[1][1]:
            raise ValueError('The last dimension of inputs[0] should match the dimension 1 of inputs[1]')

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('A `MatMul` layer should be called '
                             'on a list of inputs.')
        return tf.matmul(inputs[0], inputs[1])

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0][0], input_shape[0][1], input_shape[1][-1]]
        return tuple(output_shape)

# ------------------------------------ Pointnet Architecture
# input_Transformation_net
shap=[1,2,3,4,5,6]
input_points = Input(shape=(num_points,channel ))#
x = Convolution1D(64, 1, activation='relu',
                  input_shape=(num_points, channel),data_format='channels_last')(input_points)#
x = BatchNormalization()(x)#
if(1 in shap):
  x = Convolution1D(128, 1, activation='relu',data_format='channels_last')(x)
  x = BatchNormalization()(x)
  x = Convolution1D(512, 1, activation='relu',data_format='channels_last')(x)
  x = BatchNormalization()(x)
x = MaxPooling1D(pool_size=num_points,data_format='channels_last')(x)#
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)#
feat1=256
x = Dense(feat1, activation='relu')(x)#
x = BatchNormalization()(x)#
x = Dense(channel*channel, weights=[np.zeros([feat1, channel*channel]), np.eye(channel).flatten().astype(np.float32)])(x)#
input_T = tf.keras.layers.Reshape((channel, channel))(x)#

# forward net
if(sys.version_info[0]>=3):
  g = mat_mul(input_points,input_T)# it made model too deep to train small dataset It's better not to use now
else:
  g = Lambda(mat_mul, arguments={'B': input_T})(input_points)# it made model too deep to train small dataset It's better not to use now
#g = input_points
#g = MatMul()([input_points,input_T])#
if(2 in shap):
  g = Convolution1D(64, 1, input_shape=(num_points, channel), activation='relu',data_format='channels_last')(g)#3
  g = BatchNormalization()(g)#3
chfeat=64
g = Convolution1D(chfeat, 1, input_shape=(num_points, channel), activation='relu',data_format='channels_last')(g)#
g = BatchNormalization()(g)#

# feature transform net
ftfeat=64
f = Convolution1D(ftfeat, 1, activation='relu',data_format='channels_last')(g)#
feat3=32
if(3 in shap):
  f = BatchNormalization()(f)#3
  f = Convolution1D(128, 1, activation='relu',data_format='channels_last')(f)#3
  f = BatchNormalization()(f)#3
  f = Convolution1D(512, 1, activation='relu',data_format='channels_last')(f)#3
  f = BatchNormalization()(f)#3
f = MaxPooling1D(pool_size=num_points,data_format='channels_last')(f)#
if(4 in shap):
  f = Dense(512, activation='relu')(f)#3
  f = BatchNormalization()(f)#3
  f = Dense(256, activation='relu')(f)#3
f = BatchNormalization()(f)#

f = Dense(chfeat * chfeat, weights=[np.zeros([256, chfeat * chfeat]), np.eye(chfeat).flatten().astype(np.float32)])(f)#fit np.zeros([nodes,ch*ch]) with last layer nodes
feature_T = Reshape((chfeat, chfeat))(f)#

# forward net
if(sys.version_info[0]>=3):
  g = mat_mul(g, feature_T)#
else:
  g = Lambda(mat_mul, arguments={'B': feature_T})(g)#
#g= MatMul()([g,feature_T])#
g = Convolution1D(64, 1, activation='relu',data_format='channels_last')(g)#
g = BatchNormalization()(g)#
feat5=32
g = Convolution1D(128, 1, activation='relu',data_format='channels_last')(g)
g = BatchNormalization()(g)
g = Convolution1D(512, 1, activation='relu',data_format='channels_last')(g)
g = BatchNormalization()(g)

# global_feature
global_feature = Flatten()(MaxPooling1D(pool_size=num_points,data_format='channels_last')(g))#

# point_net_cls
c = Dense(256, activation='relu')(global_feature)#
c = BatchNormalization()(c)#
#c= Dropout(rate=0.7)(c)
if(6 in shap):
  c = Dense(256, activation="relu")(c)#2 validation increase
  c = BatchNormalization()(c)#2 0.63
#c= Dropout(rate=0.7)(c)
c = Dense(k, activation='softmax',name="output1")(c)#
# --------------------------------------------------end of pointnet

# print the model summary
#model = Model(inputs=input_points, outputs=prediction)
model = Model(inputs=input_points, outputs=c)
if(0):#still rnn works better
  gru=GRU(units=64,dropout=0.,return_sequences=False)(input_points)
  gru=Dense(64,activation='relu')(gru)
  gru=Dense(2,activation='softmax',name='output1')(gru)
  model = Model(inputs=input_points,outputs=gru)

print(model.summary())

if(sys.version_info[0]>=3):
  model_metrics=['accuracy',keras.metrics.AUC()]
else:
  model_metrics=['accuracy']
model.compile(loss=losses,
              optimizer=opt, loss_weights=lossweight,
	      metrics=model_metrics)
"""model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])
"""
savename='save/'+str(args.save)
os.makedirs(savename,exist_ok=True)
if os.path.isfile(savename+'/log.log'):
  os.remove(savename+'/log.log')
shutil.copy("symbols/symbols.py",savename+'/')
shutil.copy(__file__,savename+'/')
#plot_model(model,to_file=savename+'/model.png')
print("### plot done ###")
import logging
logging.basicConfig(filename=savename+'/log.log',level=logging.DEBUG)
logging.info(str(args))
logging.info(str(datetime.datetime.now()))
print(str(datetime.datetime.now()))
#loaded=np.load("/hdfs/store/user/yulee/DRsim/side{}pnt.npz".format(args.pt))
if("gen" in args.memo):
  loaded=np.load("gendr200.npz")
  pntset=loaded["seqset"]
  labelset=loaded["labelset"]
  X=pntset[0][:,:,1:3]# [pt,eta,phi,charge,pid]
  Y=labelset[0]
else:
  if(args.order==1):loaded=np.load("/home/yulee/keras/pntsort{}pnt{}.npz".format(args.pt,num_points))
  #if(args.order==1):loaded=np.load("/home/yulee/keras/order{}pnt{}.npz".format(args.pt,num_points))
  else:loaded=np.load("/home/yulee/keras/orgin{}pnt{}.npz".format(args.pt,num_points))
  #pntset=loaded["pntset"].item()
  pntset=loaded
  if("qg" in args.memo):
    el=pntset["uj"][:,:num_points,:4]
    pi=pntset["gj"][:,:num_points,:4]
    """el2=[]
    for i in range(len(el)/2):
      el2.append(el[2*i]+el[2*i+1])
    pi2=[]
    for i in range(len(pi)/2):
      pi2.append(pi[2*i]+pi[2*i+1])
    el=np.array(el2)
    pi=np.array(pi2)"""
  if("ep" in args.memo):
    el=pntset["el"]
    pi=pntset["pi"]
  ellabel=len(el)*[[1.,0.]]
  pilabel=len(pi)*[[0.,1.]]
  print("/hdfs/store/user/yulee/DRsim/order{}pnt.npz was loaded".format(args.pt))
  X,Y=shuffle(np.concatenate([el,pi]),np.concatenate([ellabel,pilabel]))
testX=X[int(0.7*len(X)):]
testY=Y[int(0.7*len(Y)):]
X=X[:int(0.7*len(X))]
Y=Y[:int(0.7*len(Y))]
del el
del pi
print("shape",Y.shape,X.shape)
checkpoint=keras.callbacks.ModelCheckpoint(filepath=savename+'/check_{epoch}',monitor='val_loss',verbose=0,save_weights_only=False,save_best_only=False,mode='auto',save_freq="epoch")
#for i in range(1,epochs):
#  if i % 1 == 0:
#    verbose=1
#  else:
#    verbose=0
#  if(args.rot):
#    train_points_rotate = rotate_point_cloud(X)
#    model.fit(train_points_rotate, Y, batch_size=batch_size, epochs=1, validation_split=0.3, verbose=verbose,callbacks=[checkpoint])
#  else:
#    model.fit(X, Y, batch_size=batch_size, epochs=1, validation_split=0.3, verbose=verbose,callbacks=[checkpoint])
#  #train_points_jitter = jitter_point_cloud(train_points_rotate)
#  #model.fit(train_points_jitter, Y, batch_size=64, epochs=1, shuffle=True, verbose=1)
#  #model.fit(X, Y, batch_size=128, epochs=1, shuffle=True, validation_split=0.3 verbose=1)
#  s = "Current epoch is:" + str(i)
#  print(s)
history=model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_split=0.3, verbose=1,callbacks=[checkpoint])

f=open(savename+'/history','w')
#try:
if(1):
  one=history.history['val_loss'].index(min(history.history['val_loss']))
  f.write(str(one)+'\n')
  print(one)
  for i in range(epochs):
    if(i!=one):shutil.rmtree(savename+"/check_"+str(i+1),ignore_errors=True)
  #model=keras.models.load_model(savename+"/model.h5")
  model=keras.models.load_model(savename+"/check_"+str(one+1))
  print("epoch {} loaded".format(one+1))
# score the model
#model.fit(X, Y, batch_size=128, epochs=epochs, validation_split=0.3, verbose=1)
score = model.evaluate(testX, testY, verbose=1)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
bp=model.predict(testX,verbose=0)
print("AUC:{}".format(round(roc_auc_score(testY[:,0],bp[:,0]),4)))
f=open("/home/yulee/keras/{}.auc".format(args.memo),"a")
f.write("{}".format(roc_auc_score(testY[:,0],bp[:,0])))
f.write("\n")
f.close()
np.savez("/home/yulee/keras/drbox/{}out".format(args.memo),y=testY,p=bp)

"""
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
f=open("dr1","a")
f.write("{}".format(roc_auc_score(testY[:,0],bp[:,0])))
f.write("\n")
f.close()
label="AUC:{}".format(round(roc_auc_score(testY[:,0],bp[:,0]),4))
#label="AUC:{}".format(round(roc_auc_score(Y[int(0.6*len(Y)):][:,0],bp[:,0]),4))
plt.plot(tpr,tnr,lw=3.5,label=label,linestyle="-")
plt.legend(loc=3, fontsize=fs*0.9)
plt.grid(alpha=0.6)
plt.axis((0,1,0,1))
plt.savefig("{}.png".format(args.memo),bbox_inches='tight',pad_inches=0.5)
   """   
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
  iii=1
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
    f=open("/hdfs/store/user/yulee/keras/mergelog","a")
    f.write(line1)
    f.write(line2)
    f.close()
print(str(datetime.datetime.now()-start))
