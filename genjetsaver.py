#!/usr/bin/python2.7
import os
import sys
import subprocess
import numpy as np
from geniter import *
#from jetiter import * #image original
from datetime import datetime
import random
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--etabin",type=float,default=1.,help='end ratio')
parser.add_argument("--pt",type=int,default=200,help='end ratio')
args=parser.parse_args()
ptmin=0
ptmax=0
now=datetime.now()
#for pt,frac in zip([100,200,500,1000],[0.2870344 , 0.35906137, 0.48486092, 0.60672219]):
#for pt in [100,200,500,1000]:
pt=args.pt
if(pt==100):
  ptmin=0.815
  ptmax=1.159
if(pt==200):
  ptmin=0.819
  ptmax=1.123
if(pt==500):
  ptmin=0.821
  ptmax=1.093
if(pt==1000):
  ptmin=0.8235
  ptmax=1.076

#tjdata="/home/yulee/keras/Data/jj_pt_{0}_{1}.root".format(pt,int(pt*1.1))
tjdata="/home/yulee/keras/genroot/gen_pp_jj_{0}.root".format(pt)
etabin=2.4
ptmin=0
ptmax=2
print(pt,tjdata)
train=jetiter([tjdata],batch_size=128,istrain=1,rc="r",etabin=etabin,pt=pt,ptmin=ptmin,ptmax=ptmax,unscale=1,end=1,channel=128)
#train=jetiter([tjdata],batch_size=128,istrain=1,rc="rc",etabin=etabin,pt=pt,ptmin=ptmin,ptmax=ptmax,unscale=1,end=1,channel=128)# image include
ptset=train.ptset
etaset=train.etaset
phiset=train.phiset
pidset=train.pidset
bdtset=train.bdtset #(ent,jet1+jet2)
seqset=train.seqset #(jet1+jet2,ent,pf,info)
imgset=train.imgset
labelset=train.labelset
eveset=train.eveset
pairlist=train.pairlist
#np.savez_compressed("/home/yulee/keras/gendr{}".format(pt),ptset=ptset,etaset=etaset,pidset=pidset,seqset=seqset,imgset=imgset,bdtset=bdtset,labelset=labelset,eveset=eveset,pairlist=pairlist)
np.savez_compressed("/home/yulee/keras/gendr128{}".format(pt),ptset=ptset,etaset=etaset,phiset=phiset,pidset=pidset,seqset=seqset,imgset=imgset,bdtset=bdtset,labelset=labelset,eveset=eveset,pairlist=pairlist)
print("jj128{}".format(pt))
label1=[]
label2=[]
Y=eveset
for i in range(len(Y)):
  if(Y[i][0]==1):
    label1.append([1,0])
    label2.append([1,0])
  elif(Y[i][1]==1):
    label1.append([1,0])
    label2.append([0,1])
  elif(Y[i][2]==1):
    label1.append([0,1])
    label2.append([1,0])
  elif(Y[i][3]==1):
    label1.append([0,1])
    label2.append([0,1])
print(labelset[0]==label1)
print(labelset[1]==label2)
del train
"""
zjsetz=trainz.qjetset
zptz=trainz.qptset
zetaz=trainz.qetaset
zptdz=trainz.qptdset
zchadmultz=trainz.qchadmultset
znhadmultz=trainz.qnhadmultset
zelectronmultz=trainz.qelectronmultset
zmuonmultz=trainz.qmuonmultset
zphotonmultz=trainz.qphotonmultset
zcmultz=trainz.qcmultset
znmultz=trainz.qnmultset
zmajoraxisz=trainz.qmajorset
zminoraxisz=trainz.qminorset

zjset=train.qjetset
zpt=train.qptset
zeta=train.qetaset
zptd=train.qptdset
zchadmult=train.qchadmultset
znhadmult=train.qnhadmultset
zelectronmult=train.qelectronmultset
zmuonmult=train.qmuonmultset
zphotonmult=train.qphotonmultset
zcmult=train.qcmultset
znmult=train.qnmultset
zmajoraxis=train.qmajorset
zminoraxis=train.qminorset
jjset=train.gjetset
jpt=train.gptset
jeta=train.getaset
jptd=train.gptdset
jchadmult=train.gchadmultset
jnhadmult=train.gnhadmultset
jelectronmult=train.gelectronmultset
jmuonmult=train.gmuonmultset
jphotonmult=train.gphotonmultset
jcmult=train.gcmultset
jnmult=train.gnmultset
jmajoraxis=train.gmajorset
jminoraxis=train.gminorset
ji=0
zi=0
ziz=0
lji=len(jpt)
lzi=len(zpt)
lziz=len(zptz)
ptset=[]
etaset=[]
bdtset=[]
cnnset=[]
label=[]
for i in xrange(lji+lzi+lziz):
  button=random.random()
  if(button<0.5):
    if(random.random()<frac):
      if(zi==lzi):break
      ptset.append(zpt[zi])
      etaset.append(zeta[zi])
      bdtset.append([zcmult[zi],znmult[zi],zptd[zi],zmajoraxis[zi],zminoraxis[zi],zchadmult[zi],znhadmult[zi],zelectronmult[zi],zmuonmult[zi],zphotonmult[zi]])
      cnnset.append(zjset[zi])
      label.append([0,1])
      zi+=1
    else:
      if(ji==lji):break
      ptset.append(jpt[ji])
      etaset.append(jeta[ji])
      bdtset.append([jcmult[ji],jnmult[ji],jptd[ji],jmajoraxis[ji],jminoraxis[ji],jchadmult[ji],jnhadmult[ji],jelectronmult[ji],jmuonmult[ji],jphotonmult[ji]])
      cnnset.append(jjset[ji])
      label.append([0,1])
      ji+=1
  else:
    if(ziz==lziz):break
    ptset.append(zptz[ziz])
    etaset.append(zetaz[ziz])
    bdtset.append([zcmultz[ziz],znmultz[ziz],zptdz[ziz],zmajoraxisz[ziz],zminoraxisz[ziz],zchadmultz[ziz],znhadmultz[ziz],zelectronmultz[ziz],zmuonmultz[ziz],zphotonmultz[ziz]])
    cnnset.append(zjsetz[ziz])
    label.append([1,0])
    ziz+=1
bdtset=np.array(bdtset)
cnnset=np.array(cnnset)
if(etabin==1):
  np.savez_compressed("qqggmixed{}pteta".format(pt),ptset=ptset,etaset=etaset,bdtset=bdtset,cnnset=cnnset,label=label)
if(etabin==2.4):
  np.savez_compressed("qqggmixed{}pt".format(pt),ptset=ptset,etaset=etaset,bdtset=bdtset,cnnset=cnnset,label=label)
del train
print("zi ji",zi,ji,ziz,lzi,lji,lziz)
print("name",trainz.qname,trainz.gname)
print("frac",pt,frac)
"""
