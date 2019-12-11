import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import subprocess
import numpy as np
import datetime
import random
import warnings
import ROOT as rt
import math
from array import array
from sklearn import preprocessing

class jetiter(object):
  def __init__(self,data_path,data_names=['data'],label_names=['softmax_label'],batch_size=100,begin=0.0,end=1.0,rat=0.7,endcut=1,arnum=16,maxx=0.4,maxy=0.4,istrain=0, varbs=0,rc="rc",onehot=0,channel=64,order=1,eta=0.,etabin=2.4,pt=None,ptmin=0.,ptmax=2.,unscale=1,normb=10,stride=2):
    self.eta=eta
    self.pt=pt
    self.ptmin=ptmin
    self.ptmax=ptmax
    self.etabin=etabin
    self.channel=channel
    self.istrain=istrain
    self.unscale=unscale
    self.scale=1-unscale
    self.normb=normb*1.
    self.rand=0.5
    self.count=0
    self.rc=rc
    self.onehot=onehot
    self.order=1
    dataname1=data_path[0]
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
    self.qjet.GetEntry(self.b)
    event=self.qjet.event
    self.qjet.GetEntry(self.b+1)
    if(event!=self.qjet.event):
      self.b=self.b+1
    self.ratt=rat
    self.rat=sorted([1-rat,rat])
    self.batch_size = batch_size
    """if(varbs==0):
      self._provide_data = zip(data_names, [(self.batch_size, 3, 25, 33)])
    else:
      data_names=['images','variables']
      self._provide_data = zip(data_names, [(self.batch_size, 3, 25, 33),(self.batch_size,5)])
    self.varbs=varbs
    self._provide_label = zip(label_names, [(self.batch_size,)])
    """
    self.arnum=arnum
    self.maxx=maxx
    self.maxy=maxy
    self.endfile=0
    self.endcut=endcut
    ptset=[]
    etaset=[]
    phiset=[]
    pidset=[]
    imgset=[]
    seqset=[]
    labelset=[]
    eveset=[]
    bdtset=[]
    pidic={0.0:1, 22.0: 2, 11.0: 3, -11.0: 4, 13.0: 5, -13.0: 6, 130.0: 7, 211.0: 8, -211.0: 9,  321.0: 10, -321.0: 11, 2112.0: 12, -2112.0: 13, 2212.0: 14, -2212.0: 15}
    for i in range(stride):
      ptset.append([])
      etaset.append([])
      phiset.append([])
      pidset.append([])
      imgset.append([])
      seqset.append([])
      labelset.append([])
    count=0
    count2=0
    count3=0
    event=0
    pairlist=list(self.makepair(stride=stride))
    pairset={}
    for i in range(len(pairlist)):
      card=[0]*pow(2,stride)
      card[i]=1
      pairset[pairlist[i]]=card
    qgto10={"q":[1,0],"g":[0,1]}

    self.pairset=pairset
    self.pairlist=pairlist
    for i in xrange(int(1.*(self.qEnd-self.qBegin)/stride)):
      if(self.b>=self.qEnd):
        self.b=self.qBegin
        break
      #if((self.b-self.qBegin)%int((self.qEnd-self.qBegin)/self.normb==0):print(',')
      pair=""
      pts=[]
      etas=[]
      phis=[]
      pids=[]
      imgs=[]
      seqs=[]
      bdts=[]
      for ii in range(stride):
        self.qjet.GetEntry(self.b+ii)
        if(self.eta>abs(self.qjet.eta) or self.eta+self.etabin<abs(self.qjet.eta)):
          break 
        if(self.pt!=None):
          if(self.pt*self.ptmin>self.qjet.pt or self.pt*self.ptmax<self.qjet.pt):
            break
        if(self.qjet.parton_id==0):
          break
        elif(self.qjet.parton_id==21):
          pair+="g"
        else:
          pair+="q"
        # qq 3 qg 1 gq 2 gg 0
        if(ii==0):
          event=self.qjet.event
        else:
          if(event!=self.qjet.event):
            break
        pts.append(self.qjet.pt)
        etas.append(self.qjet.eta)
        phis.append(self.qjet.phi)
        pids.append(self.qjet.parton_id)
        bdts.append(self.qjet.chad_mult+self.qjet.electron_mult+self.qjet.muon_mult)
        bdts.append(self.qjet.nhad_mult+self.qjet.photon_mult)
        bdts.append(self.qjet.ptd)
        bdts.append(self.qjet.major_axis)
        bdts.append(self.qjet.minor_axis)
        if("c" in self.rc):
          imcmult=[0.]*(625)
          imnmult=[0.]*(625)
          imcpt=[0.]*(625)
          imnpt=[0.]*(625)
          for j in range(len(self.qjet.dau_pt)):
            """if(abs(self.qjet.dau_deta[j])>0.3):continue
            if(abs(self.qjet.dau_dphi[j])>0.3):continue
            etabin=25*(self.qjet.dau_deta[j]+0.3)/(2*0.3)
            phibin=25*(self.qjet.dau_dphi[j]+0.3)/(2*0.3)
            pix=25*int(etabin)+int(phibin)
            if(self.qjet.dau_charge==0):
              imnmult[pix]=imnmult[pix]+1.
              imnpt[pix]=imnpt[pix]+self.qjet.dau_pt[j]
            else:
              imcmult[pix]=imcmult[pix]+1.
              imcpt[pix]=imcpt[pix]+self.qjet.dau_pt[j]
            #img.reshape((55,72))
          imgs=[imcmult,imnmult,imcpt,imnpt]"""
          #print(np.array(self.qjet.image_nhad_mult_25).max())
          imgs.append([np.array(self.qjet.image_chad_pt_25),np.array(self.qjet.image_nhad_pt_25),np.array(self.qjet.image_electron_pt_25),np.array(self.qjet.image_muon_pt_25),np.array(self.qjet.image_photon_pt_25),np.array(self.qjet.image_chad_mult_25),np.array(self.qjet.image_nhad_mult_25),np.array(self.qjet.image_electron_mult_25),np.array(self.qjet.image_muon_mult_25),np.array(self.qjet.image_photon_mult_25)])
        if("r" in self.rc):
          dau_pt=self.qjet.dau_pt
          dau_deta=self.qjet.dau_deta
          dau_dphi=self.qjet.dau_dphi
          dausort=sorted(range(len(dau_pt)),key=lambda k: pow(dau_deta[k],2)+pow(dau_dphi[k],2),reverse=False)
          #dausort=range(len(dau_pt))
          #dausort=sorted(range(len(dau_pt)),key=lambda k: dau_pt[k],reverse=True)
          dau_pid=self.qjet.dau_pid
          for j in range(len(dau_deta)):
            dau_deta[j]=self.qjet.dau_deta[j]+self.qjet.eta
            dau_dphi[j]=self.qjet.dau_dphi[j]+self.qjet.phi
            dau_pid[j]=pidic[dau_pid[j]]
          dau_charge=self.qjet.dau_charge
          maxdaupt=1.
          maxdaudeta=1.
          maxdaudphi=1.
          maxdaucharge=1.
          maxdauc=1.
          maxdaun=1.
          maxdaue=1.
          maxdaum=1.
          maxdaup=1.
          if(self.scale==1):
            daus=preprocessing.normalize([[dau_pt[dausort[j]]/maxdaupt, dau_deta[dausort[j]]/maxdaudeta, dau_dphi[dausort[j]]/maxdaudphi, dau_charge[dausort[j]]/maxdaucharge] if len(dau_pt)>j else [0.,0.,0.,0.] for j in range(self.channel)],axis=0)
            seqs.append(daus)
          else: 
            seqs.append([[dau_pt[dausort[j]]/maxdaupt, dau_deta[dausort[j]]/maxdaudeta, dau_dphi[dausort[j]]/maxdaudphi, dau_charge[dausort[j]]/maxdaucharge, dau_pid[dausort[j]]] if len(dau_pt)>j else [0.,0.,0.,0.,0.] for j in range(self.channel)])
      if(len(pids)==stride):
        eveset.append(pairset[pair])
        bdtset.append(bdts)
        for ii in range(stride):
          ptset[ii].append(pts[ii])
          etaset[ii].append(etas[ii])
          phiset[ii].append(phis[ii])
          pidset[ii].append(pids[ii])
          if("c" in self.rc):
            imgset[ii].append(imgs[ii])
          seqset[ii].append(seqs[ii])
          labelset[ii].append(qgto10[pair[ii]])
        count+=1
        #jetset +=...
      else:
        #print(self.b,len(pts),len(etas),len(pids))
        count3+=1
      count2+=1
      ##label q=1 g=0
      self.b+=stride
      while(self.b<self.qEnd):
        self.qjet.GetEntry(self.b)
        if(self.qjet.event==event):
          self.b+=1
        else:
          break
    self.ptset=np.array(ptset)
    del ptset
    self.etaset=np.array(etaset)
    del etaset
    self.phiset=np.array(phiset)
    del phiset
    self.pidset=np.array(pidset)
    del pidset
    self.imgset=np.array(imgset)
    del imgset
    self.seqset=np.array(seqset)
    del seqset
    self.labelset=np.array(labelset)
    del labelset
    self.eveset=np.array(eveset)
    del eveset
    self.bdtset=np.array(bdtset)
    del bdtset
    """if("r" in self.rc):
      for c in range(channel):
        for i in range(3):
          #std=np.std(abs(np.append(self.qjetset[:,c,i],self.gjetset[:,c,i])))
          #mean=np.mean(abs(np.append(self.qjetset[:,c,i],self.gjetset[:,c,i])))
          self.qjetset[:,c,i]=(self.qjetset[:,c,i])#/mean
          self.gjetset[:,c,i]=(self.gjetset[:,c,i])#/mean
    """      
    self.reset()
    print("length ",len(self.ptset),self.rc)
    print("all",count2,"pass",count,"non",count3,self.qjet.GetEntries("parton_id==0"))
  def __iter__(self):
    return self
  
  def makepair(self,pair="",stride=2,cand="qg"):
    for i in cand:
      if(stride==1):
        yield pair+i
      else:
        for loo in self.makepair(pair+i,stride-1,cand):
          yield loo

  def reset(self):
    self.rand=0.5
    self.qjet.GetEntry(self.qBegin)
    self.b=self.qBegin
    self.endfile = 0
    self.count=0

  def __next__(self):
    return self.next()

  #@property
  #def provide_data(self):
  #  return self._provide_data

  #@property
  #def provide_label(self):
  #  return self._provide_label

  def close(self):
    self.file.Close()
  def sampleallnum(self):
    return self.Entries
  def trainnum(self):
    return self.End-self.Begin
  def totalnum(self):
    return int(math.ceil(1.*(self.qEnd-self.qBegin)))
  def next(self):
    while self.endfile==0:
      self.count+=1
      arnum=self.arnum
      jetset=[]
      variables=[]
      labels=[]
      for i in range(self.batch_size):
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
