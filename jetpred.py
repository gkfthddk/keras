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
from jetiter import *
import ROOT as rt
from array import array
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
#plt.switch_backend('agg')

batch_size = 100000

parser=argparse.ArgumentParser()
parser.add_argument("--end",type=float,default=1,help='end ratio')
parser.add_argument("--epoch",type=int,default=None,help='epoch')
parser.add_argument("--save",type=str,default="test",help='rch')
parser.add_argument("--rc",type=str,default='rc',help='rnn or cnn')
parser.add_argument("--gpu",type=int,default=0,help='gpu')
parser.add_argument("--pt",type=int,default=200,help='pt range pt~pt*1.1')
parser.add_argument("--ptmin",type=float,default=0.,help='pt range pt~pt*1.1')
parser.add_argument("--ptmax",type=float,default=2.,help='pt range pt~pt*1.1')
parser.add_argument("--isz",type=int,default=0,help='0 or z or not')
parser.add_argument("--channel",type=int,default=64,help='sequence channel')
parser.add_argument("--order",type=int,default=1,help='pt ordering')
parser.add_argument("--eta",type=float,default=0.,help='end ratio')
parser.add_argument("--etabin",type=float,default=2.4,help='end ratio')
parser.add_argument("--unscale",type=int,default=1,help='end ratio')
parser.add_argument("--normb",type=float,default=10.,help='end ratio')
parser.add_argument("--stride",type=int,default=2,help='end ratio')

args=parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
config =tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.3

set_session(tf.Session(config=config))

# input image dimensions
img_rows, img_cols = 33, 33

input_shape1= (9,33,33)
input_shape2= (20,10)

#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.Adadelta(),
#              metrics=['accuracy'])
#model=keras.models.load_model('save/fullydijetsame_10')
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
vjjdata="Data/jj_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
from sklearn.metrics import roc_auc_score, auc, roc_curve
if(args.isz==0):iii=1
if(args.isz==1):iii=2
if(args.isz==-1):iii=3
#if(args.epoch==None):epoch=hist['val1_auc'.format(iii)].index(max(hist['val1_auc'.format(iii)]))+1
if(args.epoch==None):epoch=hist['val_loss'.format(iii)].index(min(hist['val_loss'.format(iii)]))+1
else:epoch=args.epoch
try:
  epoch=hist['val_loss'.format(iii)].index(min(hist['val_loss'.format(iii)]))+1
  model=keras.models.load_model(savename+"/check_"+str(epoch))
except:
  epoch=sepoch+1
  model=keras.models.load_model(savename+"/check_"+str(epoch))
rc=""
for sha in model._feed_inputs:
  if(sha._keras_shape[2]==10):
    rc+="c"
  if(sha._keras_shape[2]==64):
    rc+="r"
rc="r"
onehot=0
test2=jetiter([vjjdata],batch_size=batch_size,begin=0.7,end=1.,rc=rc,etabin=args.etabin,pt=args.pt,ptmin=args.ptmin,ptmax=args.ptmax,unscale=args.unscale,normb=args.normb,stride=args.stride)
entries=test2.totalnum()
print ("test   ",entries)
#epoch=eval(open(savename+"/history").readline())+1
#if(args.epoch==None):
f=rt.TFile("{}/get2.root".format(savename),"recreate")
#else:
#  f=rt.TFile("{}/{}get.root".format(savename,args.epoch),"recreate")
qs=[]
gs=[]
p=array('f',[0.])
pt=array('f',[0.])
eta=array('f',[0.])
pid=array('f',[0.])
trees={}
for i in range(args.stride):
  for jetid in ["q","g"]:
    trees["{}{}".format(jetid,i)]=rt.TTree("{}{}".format(jetid,i+1),"{}{} tree".format(jetid,i+1))
    trees["{}{}".format(jetid,i)].Branch("p",p,"p/F")
    trees["{}{}".format(jetid,i)].Branch("pt",pt,"pt/F")
    trees["{}{}".format(jetid,i)].Branch("eta",eta,"eta/F")
    trees["{}{}".format(jetid,i)].Branch("pid",pid,"pid/F")
#for jetid in ["qq","qg","gq","gg"]:
#    trees["{}".format(jetid)]=rt.TTree("{}".format(jetid),"{} tree".format(jetid))
#    trees["{}".format(jetid)].Branch("p",p,"p/F")
#    trees["{}".format(jetid)].Branch("pt",pt,"pt/F")
#    trees["{}".format(jetid)].Branch("eta",eta,"eta/F")
#    trees["{}".format(jetid)].Branch("pid",pid,"pid/F")
"""q1=rt.TTree("q1","q1 tree")
g1=rt.TTree("g1","g1 tree")
q2=rt.TTree("q2","q2 tree")
g2=rt.TTree("g2","g2 tree")
q1.Branch("p",p,"p/F")
q1.Branch("pt",pt,"pt/F")
q1.Branch("eta",eta,"eta/F")
q1.Branch("pid",pid,"pid/F")
g1.Branch("p",p,"p/F")
g1.Branch("pt",pt,"pt/F")
g1.Branch("eta",eta,"eta/F")
g1.Branch("pid",pid,"pid/F")
q2.Branch("p",p,"p/F")
q2.Branch("pt",pt,"pt/F")
q2.Branch("eta",eta,"eta/F")
q2.Branch("pid",pid,"pid/F")
g2.Branch("p",p,"p/F")
g2.Branch("pt",pt,"pt/F")
g2.Branch("eta",eta,"eta/F")
g2.Branch("pid",pid,"pid/F")"""
gen=test2
x=[]
y=[]
g=[]
q=[]
dens1=model.get_layer(index=21).output
dens2=model.get_layer(index=21).output
md1=keras.layers.Model(inputs=model.input,outputs=dens1)
md2=keras.layers.Model(inputs=model.input,outputs=dens2)
b1p=md1.predict(gen.seqset,verbose=0)
b2p=md2.predict(gen.seqset,verbose=0)
mb=[b1p,b2p]
bp=model.predict(gen.seqset,verbose=0)
bpt=gen.ptset
beta=gen.etaset
bpid=gen.pidset
#trees=[qs[0],qs[1],gs[0],gs[1]]
#trees=[q1,q2,g1,g2]
#qq qg gq gg
pairlist=test2.pairlist
chek=[]
for i in range(len(bp)):
  for pic in range(len(pairlist)):
    if(test2.labelset[i][pic]==1):
      for j in range(args.stride):
        p[0]=0
        for k in range(pow(2,args.stride)):
          if(pairlist[k][j]==pairlist[pic][j]):
            p[0]+=bp[i][k]
        p[0]=mb[j][i][0]
        pt[0]=bpt[i][j]
        eta[0]=beta[i][j]
        pid[0]=bpid[i][j]
        trees["{}{}".format(pairlist[pic][j],j)].Fill()
        #if(p[0]<0.2 and j == 0):
        #  #chek.append([i,test2.labelset[i],bp[i],j,p[0],pairlist[pic]])
    #p[0]=bp[i][pic]
    #pid[0]=np.where(test2.labelset[i]==1)[0][0]
    #trees["{}".format(pairlist[pic])].Fill()
"""  if(test2.labelset[i][0]==1): #qq
    for j in range(stride):
      p[0]=bp[i][0]+bp[i][1+j]
      if(j==0 and bp[i][0]+bp[i][1]!=p[0]): #qq qg
        print(i,j,test2.labelset[i])
      if(j==1 and bp[i][0]+bp[i][2]!=p[0]): #qq gq
        print(i,j,test2.labelset[i])
      pt[0]=bpt[i][j]
      eta[0]=beta[i][j]
      pid[0]=bpid[i][j]
      trees[j].Fill()
  if(test2.labelset[i][1]==1): #qg
    for j in range(2):
      p[0]=bp[i][1]+bp[i][3*j]
      if(j==0 and bp[i][1]+bp[i][0]!=p[0]):
        print(i,j,test2.labelset[i])
      if(j==1 and bp[i][1]+bp[i][3]!=p[0]):
        print(i,j,test2.labelset[i])
      pt[0]=bpt[i][j]
      eta[0]=beta[i][j]
      pid[0]=bpid[i][j]
      trees[j*3].Fill()
  if(test2.labelset[i][2]==1): #gq
    for j in range(2):
      p[0]=bp[i][2]+bp[i][3*(1-j)]
      if(j==0 and bp[i][2]+bp[i][3]!=p[0]):
        print(i,j,test2.labelset[i])
      if(j==1 and bp[i][2]+bp[i][0]!=p[0]):
        print(i,j,test2.labelset[i])
      pt[0]=bpt[i][j]
      eta[0]=beta[i][j]
      pid[0]=bpid[i][j]
      trees[2-j].Fill()
  if(test2.labelset[i][3]==1): #gg
    for j in range(2):
      p[0]=bp[i][3]+bp[i][2-j]
      if(j==0 and bp[i][3]+bp[i][2]!=p[0]):
        print(i,j,test2.labelset[i])
      if(j==1 and bp[i][3]+bp[i][1]!=p[0]):
        print(i,j,test2.labelset[i])
      pt[0]=bpt[i][j]
      eta[0]=beta[i][j]
      pid[0]=bpid[i][j]
      trees[2+j].Fill()"""
print("shape",np.array(bp).shape,test2.seqset.shape)
#for i in range(len(bp)):
#  p[0]=bp[i]
#  pt[0]=bpt[i]
#  eta[0]=beta[i]
#  pid[0]=bpid[i]
#  g1.Fill()
f.Write()
f.Close()
