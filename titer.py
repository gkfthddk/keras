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
    self.gfile=rt.TFile(dataname2,'read')
    print(dataname2)
    self.gjet=self.gfile.Get("jetAnalyser")
    #self.gim = array('B', [0]*(3*(arnum*2+1)*(arnum*2+1)))
    #self.gjet.SetBranchAddress("image", self.gim)
    #self.glabel = array('B', [0])
    #self.gjet.SetBranchAddress("label", self.glabel)
    self.gEntries=self.gjet.GetEntriesFast()
    self.gBegin=int(begin*self.gEntries)
    self.gEnd=int(self.gEntries*end)
    self.a=self.gBegin
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
    self.gjet.GetEntry(self.gBegin)
    self.a=self.gBegin
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
    return int((self.gEnd-self.gBegin)/(self.batch_size*1.00))
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
        self.gjet.GetEntry(self.a)
        dau_pt=self.gjet.dau_pt
        dau_deta=self.gjet.dau_deta
        dau_dphi=self.gjet.dau_dphi
        dau_charge=self.gjet.dau_charge
        dau_pid=self.gjet.dau_pid
        dau_is_e=np.zeros(len(dau_pid))
        dau_is_mu=np.zeros(len(dau_pid))
        dau_is_r=np.zeros(len(dau_pid))
        dau_is_chad=np.zeros(len(dau_pid))
        dau_is_nhad=np.zeros(len(dau_pid))
        for t in range(len(dau_pid)):
          if(abs(dau_pid[t])==11):dau_is_e[t]=1.
          elif(abs(dau_pid[t])==13):dau_is_mu[t]=1.
          elif(abs(dau_pid[t])==22):dau_is_r[t]=1.
          elif(dau_pid[t]==0):dau_is_nhad[t]=1.
          else:dau_is_chad[t]=1.

        self.a+=1
        if("c" in self.rc):
          #jetset.append(np.array(self.gim).reshape((3,2*arnum+1,2*arnum+1)))
          #jetset.append([np.array(self.gjet.image_chad_pt_33).reshape(2*arnum+1,2*arnum+1),np.array(self.gjet.image_nhad_pt_33).reshape(2*arnum+1,2*arnum+1),np.array(self.gjet.image_chad_mult_33).reshape(2*arnum+1,2*arnum+1)])
          jetset.append([np.array(self.gjet.image_chad_pt_33).reshape(2*arnum+1,2*arnum+1),np.array(self.gjet.image_nhad_pt_33).reshape(2*arnum+1,2*arnum+1),np.array(self.gjet.image_electron_pt_33).reshape(2*arnum+1,2*arnum+1),np.array(self.gjet.image_muon_pt_33).reshape(2*arnum+1,2*arnum+1),np.array(self.gjet.image_photon_pt_33).reshape(2*arnum+1,2*arnum+1),np.array(self.gjet.image_chad_mult_33).reshape(2*arnum+1,2*arnum+1),np.array(self.gjet.image_nhad_mult_33).reshape(2*arnum+1,2*arnum+1),np.array(self.gjet.image_electron_mult_33).reshape(2*arnum+1,2*arnum+1),np.array(self.gjet.image_muon_mult_33).reshape(2*arnum+1,2*arnum+1),np.array(self.gjet.image_photon_mult_33).reshape(2*arnum+1,2*arnum+1)])
        #label q=1 g=0
        labels.append([1-self.gjet.label,0+self.gjet.label])#q=0 q=1
        #labels.append([0+self.gjet.label,1-self.gjet.label])#q=1 g=0
        if("r" in self.rc):
          dausort=sorted(range(len(dau_pt)),key=lambda k: dau_pt[k],reverse=True)
          #dauset.append([[dau_pt[dausort[i]], dau_deta[dausort[i]], dau_dphi[dausort[i]], dau_charge[dausort[i]]] if len(dau_pt)>i else [0.,0.,0.,0.] for i in range(20)])
          dauset.append([[dau_pt[dausort[i]], dau_deta[dausort[i]], dau_dphi[dausort[i]], dau_charge[dausort[i]], dau_is_e[dausort[i]], dau_is_mu[dausort[i]], dau_is_r[dausort[i]], dau_is_chad[dausort[i]], dau_is_nhad[dausort[i]]] if len(dau_pt)>i else [0.,0.,0.,0.,0.,0.,0.,0.,0.] for i in range(20)])
        if(len(dau_pt)==0):
          print self.a
          print "@@@@@@@@@@@@@@"
          print self.a
          break
        if(self.a>=self.gEnd):
          self.a=self.gBegin
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

