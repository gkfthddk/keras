import os
import subprocess
import numpy as np
import datetime
import random
import warnings
import ROOT as rt
import math
from keras.preprocessing.sequence import pad_sequences
from array import array

class wkiter(object):
  def __init__(self,data_path,data_names=['data'],label_names=['softmax_label'],batch_size=100,begin=0.0,end=1.0,rat=0.7,endcut=1,arnum=16,maxx=0.4,maxy=0.4,istrain=0, varbs=0,rc="rc"):
    self.istrain=istrain
    #if(batch_size<100):
    self.rand=0.5
    #  print("batch_size is small it might cause error")
    self.count=0
    self.rc=rc
    #self.file=rt.TFile(data_path,'read')
    dataname1=data_path[0]
    dataname2=data_path[1]
    self.qfile=rt.TFile(dataname1,'read')
    self.gfile=rt.TFile(dataname2,'read')
    self.qjet=self.qfile.Get("image")
    self.gjet=self.gfile.Get("image")
    self.qim = array('B', [0]*(3*(arnum*2+1)*(arnum*2+1)))
    self.gim = array('B', [0]*(3*(arnum*2+1)*(arnum*2+1)))
    self.qjet.SetBranchAddress("image", self.qim)
    self.gjet.SetBranchAddress("image", self.gim)
    #self.qlabel = array('B', [0])
    #self.glabel = array('B', [0])
    #self.qjet.SetBranchAddress("label", self.qlabel)
    #self.gjet.SetBranchAddress("label", self.glabel)
    self.qEntries=self.qjet.GetEntriesFast()
    self.gEntries=self.gjet.GetEntriesFast()
    self.qBegin=int(begin*self.qEntries)
    self.gBegin=int(begin*self.gEntries)
    self.qEnd=int(self.qEntries*end)
    self.gEnd=int(self.gEntries*end)
    self.a=self.gBegin
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
  def __iter__(self):
    return self

  def reset(self):
    self.rand=0.5
    self.qjet.GetEntry(self.qBegin)
    self.gjet.GetEntry(self.gBegin)
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
    return int((self.qEnd-self.qBegin+self.gEnd-self.gBegin)/(self.batch_size*1.00))
  def next(self):
    while self.endfile==0:
      self.count+=1
      arnum=self.arnum
      jetset=[]
      dauset=[]
      variables=[]
      labels=[]
      rand=random.choice(self.rat)
      for i in range(self.batch_size):
        if(random.random()<0.5):
        #if(random.random()<rand):
          self.gjet.GetEntry(self.a)
          dau_pt=self.gjet.dau_pt
          dau_deta=self.gjet.dau_deta
          dau_dphi=self.gjet.dau_dphi
          dau_charge=self.gjet.dau_charge
          self.a+=1
          if("c" in self.rc):
            jetset.append(np.array(self.gim).reshape((3,2*arnum+1,2*arnum+1)))
          labels.append([1,0])
        else:
          self.qjet.GetEntry(self.b)
          dau_pt=self.qjet.dau_pt
          dau_deta=self.qjet.dau_deta
          dau_dphi=self.qjet.dau_dphi
          dau_charge=self.qjet.dau_charge
          self.b+=1
          if("c" in self.rc):
            jetset.append(np.array(self.qim).reshape((3,2*arnum+1,2*arnum+1)))
          labels.append([0,1])
        if("r" in self.rc):
          dausort=sorted(range(len(dau_pt)),key=lambda k: dau_pt[k],reverse=True)
          dauset.append([[dau_pt[dausort[i]], dau_deta[dausort[i]], dau_dphi[dausort[i]], dau_charge[dausort[i]]] if len(dau_pt)>i else [0.,0.,0.,0.] for i in range(20)])
        if(len(dau_pt)==0):
          print self.a, self.b
          print "@@@@@@@@@@@@@@"
          print self.a, self.b
          break
        if(self.a>=self.gEnd):
          self.a=self.gBegin
          self.endfile=1
        if(self.b>=self.qEnd):
          self.b=self.qBegin
          self.endfile=1

      data=[]
      for rc in self.rc:
        if(rc=="c"):data.append(np.array(jetset))
        if(rc=="r"):data.append(np.array(dauset))
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

