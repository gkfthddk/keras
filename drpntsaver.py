#!/usr/bin/python2.7
import os
import sys
import subprocess
import numpy as np
#from jetiter import * #image original
from datetime import datetime
import random
import argparse
import random
import warnings
import ROOT as rt
import math
from array import array
from sklearn import preprocessing
parser=argparse.ArgumentParser()
parser.add_argument("--pt",type=int,default=200,help='end ratio')
parser.add_argument("--npnt",type=int,default=1024,help='end ratio')
args=parser.parse_args()
ptmin=0
ptmax=0
now=datetime.now()
pt=args.pt
pntset={}
if(pt==20):
  namelist=["pi","el","ga"]
if(pt==50):
  #namelist=["uj","gj","pi","el","ga"]
  namelist=["uJet","gJet"]
num_pnt=args.npnt
mincount=10000
maxcount=10
minent=0
maxent=0
length=[]
for name in namelist:#,'uj','gj']:
  #infile=rt.TFile("/home/yulee/geant4/tester/analysis/{}{}sum.root".format(name,pt),'read')
  infile=rt.TFile("/home/yulee/geant4/tester/analysis/{}_{}GeV_fast.root".format(name,pt),'read')
  if(name=="uJet"):
    name="uj"
  if(name=="gJet"):
    name="gj"
  event=infile.Get("event")
  pnt_e_s=[]
  for num_entry in range(event.GetEntries()):
    if(num_entry>event.GetEntries()/2):break
    #if(num_entry>500):break
    event.GetEntry(num_entry)
    if("j" in name):
      onepnt_e2_s=[]
    onepnt_e_s=[]
    count=0
    count2=0
    if(len(event.fiber_iscerenkov)==0):continue
    for i in range(len(event.fiber_iscerenkov)):
      x=event.fiber_depth[i]
      #if(x>0 and event.fiber_ecor[i]>0.002):
      if(event.fiber_ecor[i]>0.002):
        y=event.fiber_phi[i]
        z=event.fiber_eta[i]
        jet2=0
        if("j" in name and abs(y)>rt.TMath.Pi()/2):
          jet2=1
          if(y<0):
            y=rt.TMath.Pi()+y
          else:
            y=y-rt.TMath.Pi()
        if(jet2==0):
          #if(len(onepnt_e_s)<num_pnt):
          if(not event.fiber_iscerencov[i]):
            onepnt_e_s.append([x,y,z,event.fiber_ecor[i],event.fiber_n[i],0.])
          else:
            onepnt_e_s.append([x,y,z,event.fiber_ecor[i],0.,event.fiber_n[i]])
          count+=1
        else:
          #if(len(onepnt_e2_s)<num_pnt):
          if(not event.fiber_iscerencov[i]):
            onepnt_e2_s.append([x,y,z,event.fiber_ecor[i],event.fiber_n[i],0.])
          else:
            onepnt_e2_s.append([x,y,z,event.fiber_ecor[i],0.,event.fiber_n[i]])
          count2+=1
    if(count<mincount):
      mincount=count
      minent=num_entry
    if(count>maxcount):
      maxcount=count
      maxent=num_entry
    if(count2<mincount):
      mincount=count2
      minent=num_entry
    onepnt_e_s=np.array(onepnt_e_s,dtype='float32')
    onepnt_e_s=sorted(onepnt_e_s, key=lambda pnt:pnt[3],reverse=True)
    #depth,phi,eta,energy,n
    length.append(len(onepnt_e_s))
    pnt_e_s.append([onepnt_e_s[k] if k<len(onepnt_e_s) else [0.,0.,0.,0.,0.,0] for k in range(num_pnt)])
    if("j" in name):
      #pnt_e_s.append(onepnt_e2_s)
      #pnt_e_s.append(sorted(onepnt_e2_s, key=lambda pnt:pnt[3],reverse=True)[:num_pnt])
      onepnt_e2_s=np.array(onepnt_e2_s,dtype='float32')
      onepnt_e2_s=sorted(onepnt_e2_s, key=lambda pnt:pnt[3],reverse=True)
      pnt_e_s.append([onepnt_e2_s[k] if k<len(onepnt_e2_s) else [0.,0.,0.,0.,0.,0] for k in range(num_pnt)])
  
  infile.Close()
  pntset[name]=np.array(pnt_e_s,dtype='float32')
print(pntset['uj'].shape)
print(mincount,minent)
print(maxcount,maxent)
np.savez_compressed("/home/yulee/keras/pntsort{}pnt{}".format(pt,num_pnt),uj=pntset['uj'],gj=pntset['gj'],length=length)
print(datetime.now()-now)
