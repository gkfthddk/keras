import os
import subprocess
import numpy as np
import datetime
import random
import warnings
import ROOT as rt
import math
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback
from array import array

from sklearn.metrics import roc_auc_score, auc, roc_curve


class AddVal(Callback):
  def __init__(self,valid_sets,savename):
    self.valid_sets = valid_sets
    self.epoch=[]
    self.history={}
    self.savename=savename
  
  def on_train_begin(self,logs=None):
    self.epoch=[]
    self.history={}

  def on_epoch_end(self, epoch, logs=None):
    logs=logs or {}
    self.epoch.append(epoch)

    for i,j in logs.items():
      self.history.setdefault(i,[]).append(j)

    for valid_set in self.valid_sets:
      valid,val_name=valid_set
      valid.reset()
      gen=valid.next()
      tar_set=[]
      pre_set=[]
      for j in range(valid.totalnum()):
        data,target=next(gen)
        tar_set=np.append(tar_set,target[:,0])
        pre_set=np.append(pre_set,self.model.predict(data,verbose=0)[:,0])

      valid.reset()
      auc_val=roc_auc_score(tar_set,pre_set)
      results=self.model.evaluate_generator(valid.next(),valid.totalnum())
      print(results,auc_val)

      self.history.setdefault(val_name+"_auc",[]).append(auc_val)

      for i,result in enumerate(results):
        if(i==0):
          name=val_name+"_loss"
        else:
          name=val_name+"_"+self.model.metrics[i-1][:3]
        self.history.setdefault(name,[]).append(result)
    f=open(self.savename+'/history','w')
    f.write(str(self.history))
    f.close()

class wkiter(object):
  def __init__(self,data_path,data_names=['data'],label_names=['softmax_label'],batch_size=100,begin=0.0,end=1.0,rat=0.7,endcut=1,arnum=16,maxx=0.4,maxy=0.4,istrain=0, varbs=0,rc="rc",onehot=0):
    self.istrain=istrain
    #if(batch_size<100):
    self.rand=0.5
    #  print("batch_size is small it might cause error")
    self.count=0
    self.rc=rc
    self.onehot=onehot
    #self.file=rt.TFile(data_path,'read')
    dataname1=data_path[0]
    dataname2=data_path[1]
    self.qfile=rt.TFile(dataname1,'read')
    self.gfile=rt.TFile(dataname2,'read')
    print(dataname2)
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
    return int((self.gEnd-self.gBegin+self.qEnd-self.qBegin)/(self.batch_size*1.00))
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
          self.gjet.GetEntry(self.a)
          ##label q=1 g=0
          labels.append([0,1])#q=0 q=1
          self.a+=1
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

          if("c" in self.rc):
            #jetset.append(np.array(self.gim).reshape((3,2*arnum+1,2*arnum+1)))
            #jetset.append([np.array(self.gjet.image_chad_pt_33).reshape(2*arnum+1,2*arnum+1),np.array(self.gjet.image_nhad_pt_33).reshape(2*arnum+1,2*arnum+1),np.array(self.gjet.image_chad_mult_33).reshape(2*arnum+1,2*arnum+1)])
            jetset.append([np.array(self.gjet.image_chad_pt_33).reshape(2*arnum+1,2*arnum+1),np.array(self.gjet.image_nhad_pt_33).reshape(2*arnum+1,2*arnum+1),np.array(self.gjet.image_electron_pt_33).reshape(2*arnum+1,2*arnum+1),np.array(self.gjet.image_muon_pt_33).reshape(2*arnum+1,2*arnum+1),np.array(self.gjet.image_photon_pt_33).reshape(2*arnum+1,2*arnum+1),np.array(self.gjet.image_chad_mult_33).reshape(2*arnum+1,2*arnum+1),np.array(self.gjet.image_nhad_mult_33).reshape(2*arnum+1,2*arnum+1),np.array(self.gjet.image_electron_mult_33).reshape(2*arnum+1,2*arnum+1),np.array(self.gjet.image_muon_mult_33).reshape(2*arnum+1,2*arnum+1),np.array(self.gjet.image_photon_mult_33).reshape(2*arnum+1,2*arnum+1)])
          if("r" in self.rc):
            dausort=sorted(range(len(dau_pt)),key=lambda k: dau_pt[k],reverse=True)
            #dauset.append([[dau_pt[dausort[i]], dau_deta[dausort[i]], dau_dphi[dausort[i]], dau_charge[dausort[i]]] if len(dau_pt)>i else [0.,0.,0.,0.] for i in range(20)])
            if(self.onehot):dauset.append([[dau_pt[dausort[i]], dau_deta[dausort[i]], dau_dphi[dausort[i]], dau_charge[dausort[i]], dau_pid[dausort[i]]] if len(dau_pt)>i else [0.,0.,0.,0.,0.] for i in range(20)])
            else:dauset.append([[dau_pt[dausort[i]], dau_deta[dausort[i]], dau_dphi[dausort[i]], dau_charge[dausort[i]], dau_is_e[dausort[i]], dau_is_mu[dausort[i]], dau_is_r[dausort[i]], dau_is_chad[dausort[i]], dau_is_nhad[dausort[i]]] if len(dau_pt)>i else [0.,0.,0.,0.,0.,0.,0.,0.,0.] for i in range(20)])
          if(len(dau_pt)==0):
            print self.a
            print "@@@@@@@@@@@@@@"
            print self.a
            break
          if(self.a>=self.gEnd):
            self.a=self.gBegin
            self.endfile=1
        else:
          self.qjet.GetEntry(self.b)
          ##label q=1 g=0
          labels.append([1,0])#q=0 q=1
          self.b+=1
          dau_pt=self.qjet.dau_pt
          dau_deta=self.qjet.dau_deta
          dau_dphi=self.qjet.dau_dphi
          dau_charge=self.qjet.dau_charge
          dau_pid=self.qjet.dau_pid
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

          if("c" in self.rc):
            jetset.append([np.array(self.qjet.image_chad_pt_33).reshape(2*arnum+1,2*arnum+1),np.array(self.qjet.image_nhad_pt_33).reshape(2*arnum+1,2*arnum+1),np.array(self.qjet.image_electron_pt_33).reshape(2*arnum+1,2*arnum+1),np.array(self.qjet.image_muon_pt_33).reshape(2*arnum+1,2*arnum+1),np.array(self.qjet.image_photon_pt_33).reshape(2*arnum+1,2*arnum+1),np.array(self.qjet.image_chad_mult_33).reshape(2*arnum+1,2*arnum+1),np.array(self.qjet.image_nhad_mult_33).reshape(2*arnum+1,2*arnum+1),np.array(self.qjet.image_electron_mult_33).reshape(2*arnum+1,2*arnum+1),np.array(self.qjet.image_muon_mult_33).reshape(2*arnum+1,2*arnum+1),np.array(self.qjet.image_photon_mult_33).reshape(2*arnum+1,2*arnum+1)])
          if("r" in self.rc):
            dausort=sorted(range(len(dau_pt)),key=lambda k: dau_pt[k],reverse=True)
            #dauset.append([[dau_pt[dausort[i]], dau_deta[dausort[i]], dau_dphi[dausort[i]], dau_charge[dausort[i]]] if len(dau_pt)>i else [0.,0.,0.,0.] for i in range(20)])
            if(self.onehot):dauset.append([[dau_pt[dausort[i]], dau_deta[dausort[i]], dau_dphi[dausort[i]], dau_charge[dausort[i]], dau_pid[dausort[i]]] if len(dau_pt)>i else [0.,0.,0.,0.,0.] for i in range(20)])
            else:dauset.append([[dau_pt[dausort[i]], dau_deta[dausort[i]], dau_dphi[dausort[i]], dau_charge[dausort[i]], dau_is_e[dausort[i]], dau_is_mu[dausort[i]], dau_is_r[dausort[i]], dau_is_chad[dausort[i]], dau_is_nhad[dausort[i]]] if len(dau_pt)>i else [0.,0.,0.,0.,0.,0.,0.,0.,0.] for i in range(20)])
          if(len(dau_pt)==0):
            print self.a
            print "@@@@@@@@@@@@@@"
            print self.a
            break
          if(self.b>=self.qEnd):
            self.b=self.qBegin
            self.endfile=1

      data=[]
      for rc in self.rc:
        if(rc=="r"):data.append(np.array(dauset))
        if(rc=="c"):data.append(np.array(jetset))
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
