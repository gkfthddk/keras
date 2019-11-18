#!/usr/bin/python2.7
import os
import sys
import subprocess
import numpy as np
from pfiter import *
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
#train=jetiter([tjdata],batch_size=128,istrain=1,rc="r",etabin=etabin,pt=pt,ptmin=ptmin,ptmax=ptmax,unscale=1,end=1)
train=jetiter([tjdata],batch_size=128,istrain=1,rc="rc",etabin=etabin,pt=pt,ptmin=ptmin,ptmax=ptmax,unscale=1,end=1,channel=128)
ptset=train.ptset
etaset=train.etaset
phiset=train.phiset
pidset=train.pidset
bdtset=train.bdtset
seqset=train.seqset
imgset=train.imgset
labelset=train.labelset
eveset=train.eveset
njset=train.njset
pairlist=train.pairlist
np.savez_compressed("/home/yulee/keras/gen{}".format(pt),ptset=ptset,etaset=etaset,phiset=phiset,pidset=pidset,seqset=seqset,imgset=imgset,bdtset=bdtset,labelset=labelset,eveset=eveset,njset=njset,pairlist=pairlist)
print("gen{}".format(pt))
label1=[]
label2=[]
Y=eveset
for i in range(len(Y)):
  if(len(pidset[i])!=njset[i]):
    print(len(pidset[i]),njset[i])
#print(labelset[0]==label1)
#print(labelset[1]==label2)
del train
