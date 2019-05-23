import os
import subprocess
import numpy as np
import datetime
import random
import warnings
import ROOT as rt
import math
from array import array

from sklearn.metrics import roc_auc_score, auc, roc_curve


class wkiter(object):
  def __init__(self,data_path,data_names=['data'],label_names=['softmax_label'],batch_size=100,begin=0.0,end=1.0,rat=0.7,endcut=1,arnum=16,maxx=0.4,maxy=0.4,istrain=0, varbs=0,rc="rc",onehot=0,channel=30,order=1,eta=0,etabin=2.4,pt=None,ptmin=0.,ptmax=2.):
    self.eta=eta
    self.pt=pt
    self.ptmin=ptmin
    self.ptmax=ptmax
    self.etabin=etabin
    self.etabin=etabin
    self.channel=channel
    self.istrain=istrain
    #if(batch_size<100):
    self.rand=0.5
    #  print("batch_size is small it might cause error")
    self.count=0
    self.rc=rc
    self.onehot=onehot
    self.order=order
    #self.file=rt.TFile(data_path,'read')
    dataname1=data_path[0]
    dataname2=data_path[1]
    self.qfile=rt.TFile(dataname1,'read')
    self.gfile=rt.TFile(dataname2,'read')
    print(dataname1,dataname2)
    self.gjet=self.gfile.Get("jetAnalyser")
    self.gEntries=self.gjet.GetEntriesFast()
    if(begin>1):
      self.gBegin=int(begin)
    else:
      self.gBegin=int(begin*self.gEntries)
    if(end>1):
      self.gEnd=int(end)
    else: 
      self.gEnd=int(self.gEntries*end)
    self.a=self.gBegin
    self.qjet=self.qfile.Get("jetAnalyser")
    self.qEntries=self.qjet.GetEntriesFast()
    if(begin>1):
      self.qBegin=int(begin)
    else:
      self.qBegin=int(begin*self.qEntries)
    if(end>1):
      self.qEnd=int(end)
    else: 
      self.qEnd=int(self.qEntries*end)
    self.b=self.qBegin
    self.ratt=rat
    self.rat=sorted([1-rat,rat])
    self.batch_size = batch_size
    if(varbs==0):
      self._provide_data = zip(data_names, [(self.batch_size, 3, 33, 33)])
    else:
      data_names=['images','variables']
      self._provide_data = zip(data_names, [(self.batch_size, 3, 33, 33),(self.batch_size,5)])
    self.varbs=varbs
    self._provide_label = zip(label_names, [(self.batch_size,)])
    self.arnum=arnum
    self.maxx=maxx
    self.maxy=maxy
    self.endfile=0
    self.endcut=endcut
    qjetset=[]
    gjetset=[]
    qptset=[]
    gptset=[]
    qetaset=[]
    getaset=[]
    for i in range(self.gEntries):
      if(self.a>=self.gEnd):
        self.a=self.gBegin
        break
      #if((self.a-self.gBegin)%int((self.gEnd-self.gBegin)/10)==0):print('.')
      self.gjet.GetEntry(self.a)
      ##label q=1 g=0
      self.a+=1
      if(self.eta>abs(self.gjet.eta) or self.eta+self.etabin<abs(self.gjet.eta)):
        continue
      if(self.pt!=None):
        if(self.pt*self.ptmin>self.gjet.pt or self.pt*self.ptmax<self.gjet.pt):
          continue
      gptset.append(self.gjet.pt)
      getaset.append(self.gjet.eta)
      #gjetset.append([self.gjet.pt,self.gjet.eta,self.gjet.phi,self.gjet.ptd,self.gjet.major_axis,self.gjet.minor_axis,self.gjet.chad_mult,self.gjet.nhad_mult,self.gjet.electron_mult,self.gjet.muon_mult,self.gjet.photon_mult])
      gjetset.append([self.gjet.ptd,self.gjet.major_axis,self.gjet.minor_axis,self.gjet.chad_mult+self.gjet.electron_mult+self.gjet.muon_mult,self.gjet.nhad_mult+self.gjet.photon_mult])
    self.gjetset=np.array(gjetset)
    del gjetset
    self.gptset=np.array(gptset)
    del gptset
    self.getaset=np.array(getaset)
    del getaset
    for i in range(self.qEntries):
      if(self.b>=self.qEnd):
        self.b=self.qBegin
        break
      #if((self.b-self.qBegin)%int((self.qEnd-self.qBegin)/10)==0):print(',')
      self.qjet.GetEntry(self.b)
      ##label q=1 g=0
      self.b+=1
      if(self.eta>abs(self.qjet.eta) or self.eta+self.etabin<abs(self.qjet.eta)):
        continue
      if(self.pt!=None):
        if(self.pt*self.ptmin>self.qjet.pt or self.pt*self.ptmax<self.qjet.pt):
          continue
      qptset.append(self.qjet.pt)
      qetaset.append(self.qjet.eta)
      #qjetset.append([self.qjet.pt,self.qjet.eta,self.qjet.phi,self.qjet.ptd,self.qjet.major_axis,self.qjet.minor_axis,self.qjet.chad_mult,self.qjet.nhad_mult,self.qjet.electron_mult,self.qjet.muon_mult,self.qjet.photon_mult])
      qjetset.append([self.qjet.ptd,self.qjet.major_axis,self.qjet.minor_axis,self.qjet.chad_mult+self.qjet.electron_mult+self.qjet.muon_mult,self.qjet.nhad_mult+self.qjet.photon_mult])
    self.qjetset=np.array(qjetset)
    del qjetset
    self.qptset=np.array(qptset)
    del qptset
    self.qetaset=np.array(qetaset)
    del qetaset
    print(self.qjetset.shape)
    self.reset()
  def __iter__(self):
    return self

  def reset(self):
    self.rand=0.5
    self.gjet.GetEntry(self.gBegin)
    self.qjet.GetEntry(self.qBegin)
    self.a=self.gBegin
    self.b=self.qBegin
    self.endfile = 0
    self.count=0

  def __next__(self):
    return self.next()

  @property
  def provide_data(self):
    return self._provide_data

  @property
  def provide_label(self):
    return self._provide_label

  def close(self):
    self.file.Close()
  def sampleallnum(self):
    return self.Entries
  def trainnum(self):
    return self.End-self.Begin
  def totalnum(self):
    return int(1.*(self.gEnd-self.gBegin+self.qEnd-self.qBegin)/(self.batch_size*1.00))
  def next(self):
    while self.endfile==0:
      self.count+=1
      arnum=self.arnum
      jetset=[]
      variables=[]
      labels=[]
      for i in range(self.batch_size):
        if(random.random()<0.5):
          #if(abs(self.gjetset[self.a-self.gBegin][1])>self.eta+self.etabin or abs(self.gjetset[self.a-self.gBegin][1])<self.eta):
          #  self.a+=1
          #  continue
          if(self.a-self.gBegin>=len(self.gjetset)):
            self.a=self.gBegin
            self.endfile=1
            break
          labels.append([0,1])
          jetset.append(self.gjetset[self.a-self.gBegin])
          self.a+=1
        else:
          #if(abs(self.qjetset[self.b-self.qBegin][1])>self.eta+self.etabin or abs(self.qjetset[self.b-self.qBegin][1])<self.eta):
          #  self.b+=1
          #  continue
          if(self.b-self.qBegin>=len(self.qjetset)):
            self.b=self.qBegin
            self.endfile=1
            break
          labels.append([1,0])
          jetset.append(self.qjetset[self.b-self.qBegin])
          self.b+=1
      data=[]
      data.append(np.array(jetset))
      label=np.array(labels)
      #if(self.totalnum()<=self.count):
      #  if(self.istrain==1):print "\nreset\n"
      #  self.reset()
      if(self.endfile==1):
        #print "\nendd\n"
        self.reset()
      #print "\n",self.count,self.istrain,"\n"
      yield data, label
      #else:
        #if(self.istrain==1):
        #  print "\n",datetime.datetime.now()  
        #raise StopIteration
