#!/usr/bin/python2.7
from __future__ import print_function
import argparse
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras
from keras import backend as K
import ROOT as rt
from array import array
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
#python jetdualpred.py --save dualn2500 --pt 500 --stride 2 --gpu 3
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
parser.add_argument("--mod",type=int,default=0,help='end ratio')

args=parser.parse_args()
if(args.gpu!=-1):
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
  if(sha._keras_shape[-1]==33*33):
    rc+="c"
  if(sha._keras_shape[-1]==33):
    rc+="c"
onehot=0
loaded=np.load("pf{}.npz".format(args.pt))
if("c" in rc):
  X=loaded["imgset"]
else:
  X=loaded["seqset"][:,:,:4]
Y=loaded["labelset"]
X=X[80000:101000]
Y=Y[80000:101000]
#epoch=eval(open(savename+"/history").readline())+1
#if(args.epoch==None):
f=rt.TFile("{}/getd.root".format(savename),"recreate")
#else:
#  f=rt.TFile("{}/{}get.root".format(savename,args.epoch),"recreate")
qs=[]
gs=[]
p=array('f',[0.])
pt=array('f',[0.])
eta=array('f',[0.])
pid=array('f',[0.])
trees={}
for i in range(2):
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
label1=Y[:,0]
label2=Y[:,1]
if(args.stride==1):
  X=X.reshape((-1,10,33*33))
  Y=Y.reshape((-1,2))
x=[]
y=[]
g=[]
q=[]
bp=model.predict(X,verbose=0)
bpt=loaded["ptset"][80000:101000]
beta=loaded["etaset"][80000:101000]
bpid=loaded["pidset"][80000:101000]
#trees=[qs[0],qs[1],gs[0],gs[1]]
#trees=[q1,q2,g1,g2]
#qq qg gq gg
chek=[]
"""if(args.stride==1):
  for i in range(int(len(bp)/2)):
    if(Y[i][0]==1):
      p[0]=bp[i]
      pt[0]=bpt[0][i]
      eta[0]=beta[0][i]
      pid[0]=bpid[0][i]
      trees["{}{}".format("q",0)].Fill()
    if(Y[i][0]==0):
      p[0]=bp[i]
      pt[0]=bpt[1][i]
      eta[0]=beta[1][i]
      pid[0]=bpid[1][i]
      trees["{}{}".format("g",1)].Fill()"""
if(args.stride==2):
  if(args.mod==0):leng= len(bp)
  else:leng=len(bp)
  for i in range(leng):
    for j in range(args.stride):
      if(label1[i][j]==1):
        if(args.mod==0):p[0]=bp[0][j][i]
        else:p[0]=bp[i][2*j]+bp[i][2*j+1]
        pt[0]=bpt[i][0]
        eta[0]=beta[i][0]
        pid[0]=bpid[i][0]
        trees["{}{}".format(["q","g"][j],0)].Fill()
      if(label2[i][j]==1):
        if(args.mod==0):p[0]=bp[1][j][i]
        else:p[0]=bp[i][j]+bp[i][2+j]
        pt[0]=bpt[i][1]
        eta[0]=beta[i][1]
        pid[0]=bpid[i][1]
        trees["{}{}".format(["q","g"][j],1)].Fill()
        #if(p[0]<0.2 and j == 0):
    #p[0]=bp[i][pic]
    #trees["{}".format(pairlist[pic])].Fill()
#for i in range(len(bp)):
#  p[0]=bp[i]
#  pt[0]=bpt[i]
#  eta[0]=beta[i]
#  pid[0]=bpid[i]
#  g1.Fill()
f.Write()
f.Close()
if(args.stride==1):
  line1="{} roc {} ".format(args.save,round(roc_auc_score(Y[:,0],bp[:,0]),5))
  print(line1)
  f=open("mergelog","a")
  f.write(line1)
if(args.stride==2):
  line1="{} roc 12 {} {} mean {} \n".format(args.save,round(roc_auc_score(label1[:,0],bp[0][:,0]),5),round(roc_auc_score(label2[:,0],bp[1][:,0]),5),round(roc_auc_score(np.concatenate([label1[:,0],label2[:,0]]),np.concatenate([bp[0][:,0],bp[1][:,0]])),5))
  score1=round(roc_auc_score(label1[:,0],bp[0][:,0]),5)
  #bp=model.predict([X[0],X[0]],verbose=0)
  #line2="{} roc 11 {} {} {} \n".format(args.save,round(roc_auc_score(label1[:,0],bp[0][:,0]),5),round(roc_auc_score(label1[:,0],bp[1][:,0]),5),score1-round(roc_auc_score(label1[:,0],bp[0][:,0]),5))
  print(line1)
  #print(line2)
  f=open("mergelog","a")
  f.write(line1)
  #f.write(line2)
  f.close()
