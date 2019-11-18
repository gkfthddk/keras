import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
    print("validation")
    for i,j in logs.items():
      self.history.setdefault(i,[]).append(j)

    for valid_set in self.valid_sets:
      valid,val_name=valid_set
      #valid.reset()
      #gen=valid.next()
      #tar_set=[]
      #pre_set=[]
      atar_set=[]
      apre_set=[]
      X,Y=valid
      #X=X[0]
      
      """for j in range(valid.totalnum()):
        data,target=next(gen)
        #print(target)
        #tar_set=np.append(tar_set,target[:,0])
        #pre_set=np.append(pre_set,self.model.predict(data,verbose=0)[:,0])
        try:atar_set.extend(target[:,0])
        except:print(np.array(target).shape)
        apre_set.extend(self.model.predict(data,verbose=0)[:,0])

      valid.reset()"""
      #tar_set=np.array(tar_set)
      #pre_set=np.array(pre_set)
      
      atar_set=np.array(Y)[:,0]
      apre_set=np.array(self.model.predict(X,verbose=0)[:,0])
      
      #print(valid.totalnum(),valid.batch_size)
      #print("############")
      #print(tar_set)
      #print("AAAAAAAAAAAAAAAAAAAA")
      #print(atar_set)

      auc_val=roc_auc_score(atar_set,apre_set)
      results=self.model.evaluate(X,Y)
      print("validation",results,auc_val)

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
  def __init__(self,data_path,data_names=['data'],label_names=['softmax_label'],batch_size=100,begin=0.0,end=1.0,rat=0.7,endcut=1,arnum=16,maxx=0.4,maxy=0.4,istrain=0, varbs=0,rc="rc",onehot=0,channel=64,order=1,eta=0.,etabin=2.4,pt=None,ptmin=0.,ptmax=2.,unscale=0,normb=10):
    self.eta=eta
    self.pt=pt
    self.ptmin=ptmin
    self.ptmax=ptmax
    self.etabin=etabin
    self.channel=channel
    self.istrain=istrain
    self.unscale=unscale
    self.normb=normb*1.
    #if(batch_size<100):
    self.rand=0.5
    #  print("batch_size is small it might cause error")
    self.count=0
    self.rc=rc
    self.onehot=onehot
    self.order=1
    #self.file=rt.TFile(data_path,'read')
    dataname1=data_path[0]
    dataname2=data_path[1]
    self.qfile=rt.TFile(dataname1,'read')
    print(dataname1)
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
    qpidset=[]
    gpidset=[]
    for i in range(self.qEntries):
      if(self.b>=self.qEnd):
        self.b=self.qBegin
        break
      #if((self.b-self.qBegin)%int((self.qEnd-self.qBegin)/self.normb==0):print(',')
      self.qjet.GetEntry(self.b)
      ##label q=1 g=0
      self.b+=1
      if(self.eta>abs(self.qjet.eta) or self.eta+self.etabin<abs(self.qjet.eta)):
        continue
      if(self.pt!=None):
        if(self.pt*self.ptmin>self.qjet.pt or self.pt*self.ptmax<self.qjet.pt):
          continue
      if(self.qjet.parton_id==21):
        gptset.append(self.qjet.pt)
        getaset.append(self.qjet.eta)
        gpidset.append(self.qjet.parton_id)
        if("c" in self.rc):
          maxchadpt=1.*max(self.qjet.image_chad_pt_33)/self.normb
          maxnhadpt=1.*max(self.qjet.image_nhad_pt_33)/self.normb
          maxelecpt=1.*max(self.qjet.image_electron_pt_33)/self.normb
          maxmuonpt=1.*max(self.qjet.image_muon_pt_33)/self.normb
          maxphotonpt=1.*max(self.qjet.image_photon_pt_33)/self.normb
          maxchadmult=1.*max(self.qjet.image_chad_mult_33)/self.normb
          maxnhadmult=1.*max(self.qjet.image_nhad_mult_33)/self.normb
          maxelecmult=1.*max(self.qjet.image_electron_mult_33)/self.normb
          maxmuonmult=1.*max(self.qjet.image_muon_mult_33)/self.normb
          maxphotonmult=1.*max(self.qjet.image_photon_mult_33)/self.normb
          if(self.unscale==1 or maxchadpt==0):maxchadpt=1.
          if(self.unscale==1 or maxnhadpt==0):maxnhadpt=1.
          if(self.unscale==1 or maxelecpt==0):maxelecpt=1.
          if(self.unscale==1 or maxmuonpt==0):maxmuonpt=1.
          if(self.unscale==1 or maxphotonpt==0):maxphotonpt=1.
          if(self.unscale==1 or maxchadmult==0):maxchadmult=1.
          if(self.unscale==1 or maxnhadmult==0):maxnhadmult=1.
          if(self.unscale==1 or maxelecmult==0):maxelecmult=1.
          if(self.unscale==1 or maxmuonmult==0):maxmuonmult=1.
          if(self.unscale==1 or maxphotonmult==0):maxphotonmult=1.
          gjetset.append([(np.array(self.qjet.image_chad_pt_33)/maxchadpt).reshape(2*arnum+1,2*arnum+1),(np.array(self.qjet.image_nhad_pt_33)/maxnhadpt).reshape(2*arnum+1,2*arnum+1),(np.array(self.qjet.image_electron_pt_33)/maxelecpt).reshape(2*arnum+1,2*arnum+1),(np.array(self.qjet.image_muon_pt_33)/maxmuonpt).reshape(2*arnum+1,2*arnum+1),(np.array(self.qjet.image_photon_pt_33)/maxphotonpt).reshape(2*arnum+1,2*arnum+1),(np.array(self.qjet.image_chad_mult_33)/maxchadmult).reshape(2*arnum+1,2*arnum+1),(np.array(self.qjet.image_nhad_mult_33)/maxnhadmult).reshape(2*arnum+1,2*arnum+1),(np.array(self.qjet.image_electron_mult_33)/maxelecmult).reshape(2*arnum+1,2*arnum+1),(np.array(self.qjet.image_muon_mult_33)/maxmuonmult).reshape(2*arnum+1,2*arnum+1),(np.array(self.qjet.image_photon_mult_33)/maxphotonmult).reshape(2*arnum+1,2*arnum+1)])
        if("r" in self.rc):
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
            elif(dau_charge[t]==0):dau_is_nhad[t]=1.
            else:dau_is_chad[t]=1.
          dausort=sorted(range(len(dau_pt)),key=lambda k: dau_pt[k],reverse=True)
          #dauset.append([[dau_pt[dausort[i]], dau_deta[dausort[i]], dau_dphi[dausort[i]], dau_charge[dausort[i]]] if len(dau_pt)>i else [0.,0.,0.,0.] for i in range(20)])
          if(self.order):
            maxdaupt=1.*max(dau_pt)/self.normb
            maxdaudeta=1.*max(dau_deta)/self.normb
            maxdaudphi=1.*max(dau_dphi)/self.normb
            maxdaucharge=1.*max(dau_charge)/self.normb
            maxdauc=1.*max(dau_is_chad)/self.normb
            maxdaun=1.*max(dau_is_nhad)/self.normb
            maxdaue=1.*max(dau_is_e)/self.normb
            maxdaum=1.*max(dau_is_mu)/self.normb
            maxdaup=1.*max(dau_is_r)/self.normb
            if(self.unscale==1 or maxdaupt==0):maxdaupt=1.
            if(self.unscale==1 or maxdaudeta==0):maxdaudeta=1.
            if(self.unscale==1 or maxdaudphi==0):maxdaudphi=1.
            if(self.unscale==1 or maxdaucharge==0):maxdaucharge=1.
            if(self.unscale==1 or maxdauc==0):maxdauc=1.
            if(self.unscale==1 or maxdaun==0):maxdaun=1.
            if(self.unscale==1 or maxdaue==0):maxdaue=1.
            if(self.unscale==1 or maxdaum==0):maxdaum=1.
            if(self.unscale==1 or maxdaup==0):maxdaup=1.
            gjetset.append([[dau_pt[dausort[i]]/maxdaupt, dau_deta[dausort[i]]/maxdaudeta, dau_dphi[dausort[i]]/maxdaudphi, dau_charge[dausort[i]]/maxdaucharge, dau_is_e[dausort[i]]/maxdaue, dau_is_mu[dausort[i]]/maxdaum, dau_is_r[dausort[i]]/maxdaup, dau_is_chad[dausort[i]]/maxdauc, dau_is_nhad[dausort[i]]/maxdaun] if len(dau_pt)>i else [0.,0.,0.,0.,0.,0.,0.,0.,0.] for i in range(self.channel)])
      elif(self.qjet.parton_id!=0):
        qptset.append(self.qjet.pt)
        qetaset.append(self.qjet.eta)
        qpidset.append(self.qjet.parton_id)
        if("c" in self.rc):
          maxchadpt=1.*max(self.qjet.image_chad_pt_33)/self.normb
          maxnhadpt=1.*max(self.qjet.image_nhad_pt_33)/self.normb
          maxelecpt=1.*max(self.qjet.image_electron_pt_33)/self.normb
          maxmuonpt=1.*max(self.qjet.image_muon_pt_33)/self.normb
          maxphotonpt=1.*max(self.qjet.image_photon_pt_33)/self.normb
          maxchadmult=1.*max(self.qjet.image_chad_mult_33)/self.normb
          maxnhadmult=1.*max(self.qjet.image_nhad_mult_33)/self.normb
          maxelecmult=1.*max(self.qjet.image_electron_mult_33)/self.normb
          maxmuonmult=1.*max(self.qjet.image_muon_mult_33)/self.normb
          maxphotonmult=1.*max(self.qjet.image_photon_mult_33)/self.normb
          if(self.unscale==1 or maxchadpt==0):maxchadpt=1.
          if(self.unscale==1 or maxnhadpt==0):maxnhadpt=1.
          if(self.unscale==1 or maxelecpt==0):maxelecpt=1.
          if(self.unscale==1 or maxmuonpt==0):maxmuonpt=1.
          if(self.unscale==1 or maxphotonpt==0):maxphotonpt=1.
          if(self.unscale==1 or maxchadmult==0):maxchadmult=1.
          if(self.unscale==1 or maxnhadmult==0):maxnhadmult=1.
          if(self.unscale==1 or maxelecmult==0):maxelecmult=1.
          if(self.unscale==1 or maxmuonmult==0):maxmuonmult=1.
          if(self.unscale==1 or maxphotonmult==0):maxphotonmult=1.
          qjetset.append([(np.array(self.qjet.image_chad_pt_33)/maxchadpt).reshape(2*arnum+1,2*arnum+1),(np.array(self.qjet.image_nhad_pt_33)/maxnhadpt).reshape(2*arnum+1,2*arnum+1),(np.array(self.qjet.image_electron_pt_33)/maxelecpt).reshape(2*arnum+1,2*arnum+1),(np.array(self.qjet.image_muon_pt_33)/maxmuonpt).reshape(2*arnum+1,2*arnum+1),(np.array(self.qjet.image_photon_pt_33)/maxphotonpt).reshape(2*arnum+1,2*arnum+1),(np.array(self.qjet.image_chad_mult_33)/maxchadmult).reshape(2*arnum+1,2*arnum+1),(np.array(self.qjet.image_nhad_mult_33)/maxnhadmult).reshape(2*arnum+1,2*arnum+1),(np.array(self.qjet.image_electron_mult_33)/maxelecmult).reshape(2*arnum+1,2*arnum+1),(np.array(self.qjet.image_muon_mult_33)/maxmuonmult).reshape(2*arnum+1,2*arnum+1),(np.array(self.qjet.image_photon_mult_33)/maxphotonmult).reshape(2*arnum+1,2*arnum+1)])
        if("r" in self.rc):
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
            elif(dau_charge[t]==0):dau_is_nhad[t]=1.
            else:dau_is_chad[t]=1.
          dausort=sorted(range(len(dau_pt)),key=lambda k: dau_pt[k],reverse=True)
          #dauset.append([[dau_pt[dausort[i]], dau_deta[dausort[i]], dau_dphi[dausort[i]], dau_charge[dausort[i]]] if len(dau_pt)>i else [0.,0.,0.,0.] for i in range(20)])
          if(self.order):
            maxdaupt=1.*max(dau_pt)/self.normb
            maxdaudeta=1.*max(dau_deta)/self.normb
            maxdaudphi=1.*max(dau_dphi)/self.normb
            maxdaucharge=1.*max(dau_charge)/self.normb
            maxdauc=1.*max(dau_is_chad)/self.normb
            maxdaun=1.*max(dau_is_nhad)/self.normb
            maxdaue=1.*max(dau_is_e)/self.normb
            maxdaum=1.*max(dau_is_mu)/self.normb
            maxdaup=1.*max(dau_is_r)/self.normb
            if(self.unscale==1 or maxdaupt==0):maxdaupt=1.
            if(self.unscale==1 or maxdaudeta==0):maxdaudeta=1.
            if(self.unscale==1 or maxdaudphi==0):maxdaudphi=1.
            if(self.unscale==1 or maxdaucharge==0):maxdaucharge=1.
            if(self.unscale==1 or maxdauc==0):maxdauc=1.
            if(self.unscale==1 or maxdaun==0):maxdaun=1.
            if(self.unscale==1 or maxdaue==0):maxdaue=1.
            if(self.unscale==1 or maxdaum==0):maxdaum=1.
            if(self.unscale==1 or maxdaup==0):maxdaup=1.
            qjetset.append([[dau_pt[dausort[i]]/maxdaupt, dau_deta[dausort[i]]/maxdaudeta, dau_dphi[dausort[i]]/maxdaudphi, dau_charge[dausort[i]]/maxdaucharge, dau_is_e[dausort[i]]/maxdaue, dau_is_mu[dausort[i]]/maxdaum, dau_is_r[dausort[i]]/maxdaup, dau_is_chad[dausort[i]]/maxdauc, dau_is_nhad[dausort[i]]/maxdaun] if len(dau_pt)>i else [0.,0.,0.,0.,0.,0.,0.,0.,0.] for i in range(self.channel)])
    self.gjetset=np.array(gjetset)
    del gjetset
    self.gptset=np.array(gptset)
    del gptset
    self.getaset=np.array(getaset)
    del getaset
    self.gpidset=np.array(gpidset)
    del gpidset
    self.qjetset=np.array(qjetset)
    del qjetset
    self.qptset=np.array(qptset)
    del qptset
    self.qetaset=np.array(qetaset)
    del qetaset
    self.qpidset=np.array(qpidset)
    del qpidset
    self.qBegin=0
    self.qEnd=len(self.qpidset)
    self.gBegin=0
    self.gEnd=len(self.gpidset)
    self.a=0
    self.b=0
    self.reset()
    print("length ",len(self.gjetset),len(self.qjetset))
  def __iter__(self):
    return self

  def reset(self):
    self.rand=0.5
    self.a=0
    self.b=0
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
    return int(math.ceil(1.*(self.gEnd-self.gBegin+self.qEnd-self.qBegin)/(self.batch_size*1.00)))
  def next(self):
    while self.endfile==0:
      self.count+=1
      arnum=self.arnum
      jetset=[]
      variables=[]
      labels=[]
      for i in range(self.batch_size):
        if(random.random()<0.5):
          if(self.a-self.gBegin>=self.gEnd):
            self.a=self.gBegin
            self.endfile=1
            break
          labels.append([0,1])
          jetset.append(self.gjetset[self.a-self.gBegin])
          self.a+=1
        else:
          if(self.b-self.qBegin>=self.qEnd):
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
