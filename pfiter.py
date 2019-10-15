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
  def __init__(self,data_path,data_names=['data'],label_names=['softmax_label'],batch_size=100,begin=0.0,end=1.0,rat=0.7,endcut=1,arnum=16,maxx=0.4,maxy=0.4,istrain=0, varbs=0,rc="rc",onehot=0,channel=100,order=1,eta=0.,etabin=2.4,pt=None,ptmin=0.,ptmax=2.,unscale=1,normb=10,stride=2):
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
      self._provide_data = zip(data_names, [(self.batch_size, 3, 33, 33)])
    else:
      data_names=['images','variables']
      self._provide_data = zip(data_names, [(self.batch_size, 3, 33, 33),(self.batch_size,5)])
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
    njset=[]
    bdtset=[]
    count=0
    count2=0
    count3=0
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
      dau_pts=[]
      dau_etas=[]
      dau_phis=[]
      dau_charges=[]
      dau_es=[]
      dau_mus=[]
      dau_rs=[]
      dau_nhads=[]
      dau_chads=[]
      self.qjet.GetEntry(self.b)
      numjet=self.qjet.num_jets
      leta=[]
      lphi=[]
      lpt=[]
      for ii in xrange(numjet):
        self.qjet.GetEntry(self.b+ii)
        if(self.eta>abs(self.qjet.eta) or self.eta+self.etabin<abs(self.qjet.eta)):
          break 
        if(ii<2):
          if(self.pt!=None):
            if(self.pt*self.ptmin>self.qjet.pt or self.pt*self.ptmax<self.qjet.pt):
              print("pt")
              break
          if(self.qjet.parton_id==0):
            break
          elif(self.qjet.parton_id==21):
            pair+="g"
          else:
            pair+="q"
          bdts.append(self.qjet.chad_mult+self.qjet.electron_mult+self.qjet.muon_mult)
          bdts.append(self.qjet.nhad_mult+self.qjet.photon_mult)
          bdts.append(self.qjet.ptd)
          bdts.append(self.qjet.major_axis)
          bdts.append(self.qjet.minor_axis)
          leta.append(self.qjet.eta)
          lphi.append(self.qjet.phi)
          lpt.append(self.qjet.pt)
        # qq 3 qg 1 gq 2 gg 0
        if(ii==0):
          event=self.qjet.event
        else:
          if(event!=self.qjet.event):
            print("event",event)
            print(self.b+ii,ii,self.qjet.event,numjet) 
            self.qjet.GetEntry(self.b+ii-1)
            print(self.b+ii-1,ii,self.qjet.event,numjet) 
            break
        pts.append(self.qjet.pt)
        etas.append(self.qjet.eta)
        phis.append(self.qjet.phi)
        pids.append(self.qjet.parton_id)
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
          for j in range(len(dau_deta)):
            dau_deta[j]=self.qjet.dau_deta[j]+self.qjet.eta
            dau_dphi[j]=self.qjet.dau_dphi[j]+self.qjet.phi
            if(abs(dau_pid[j])==11):dau_is_e[j]=1.
            elif(abs(dau_pid[j])==13):dau_is_mu[j]=1.
            elif(abs(dau_pid[j])==22):dau_is_r[j]=1.
            elif(dau_charge[j]==0):dau_is_nhad[j]=1.
            else:dau_is_chad[j]=1.
          dau_pts.extend(dau_pt)
          dau_etas.extend(dau_deta)
          dau_phis.extend(dau_dphi)
          dau_charges.extend(dau_charge)
          dau_es.extend(dau_is_e)
          dau_mus.extend(dau_is_mu)
          dau_rs.extend(dau_is_r)
          dau_nhads.extend(dau_is_nhad)
          dau_chads.extend(dau_is_chad)
      if(len(pids)==numjet):
        eveset.append(pairset[pair])
        njset.append(numjet)
        bdts.append(np.math.sqrt(pow(leta[0]-leta[1],2)+pow(lphi[0]-lphi[1],2)))
        bdts.append(lpt[0]/lpt[1])
        bdtset.append(bdts)
        labelset.append([qgto10[pair[0]],qgto10[pair[1]]])
        ptset.append(pts)
        etaset.append(etas)
        phiset.append(phis)
        pidset.append(pids)
        #dausort=range(len(dau_pts))
        if("c" in self.rc):
          imcmult=[0.]*(72*55)
          imnmult=[0.]*(72*55)
          imcpt=[0.]*(72*55)
          imnpt=[0.]*(72*55)
          for j in range(len(dau_pts)):
            if(abs(dau_etas[j])>2.4):continue
            if(abs(dau_phis[j])>3.14159):continue
            etabin=55*(dau_etas[j]+2.4)/(2*2.4)
            phibin=72*(dau_phis[j]+3.14159)/(2*3.14159)
            pix=72*int(etabin)+int(phibin)
            if(dau_charges==0):
              imnmult[pix]=imnmult[pix]+1.
              imnpt[pix]=imnpt[pix]+dau_pts[j]
            else:
              imcmult[pix]=imcmult[pix]+1.
              imcpt[pix]=imcpt[pix]+dau_pts[j]
            #img.reshape((55,72))
          imgs=[imcmult,imnmult,imcpt,imnpt]
        imgset.append(imgs)
        if("r" in self.rc):
          dausort=sorted(range(len(dau_pts)),key=lambda k: dau_pts[k],reverse=True)
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
            daus=preprocessing.normalize([[dau_pts[dausort[j]]/maxdaupt, dau_etas[dausort[j]]/maxdaudeta, dau_phis[dausort[j]]/maxdaudphi, dau_charges[dausort[j]]/maxdaucharge] if len(dau_pt)>j else [0.,0.,0.,0.] for j in range(self.channel)],axis=0)
            seqs.append(daus)
          else: 
            seqs=[[dau_pts[dausort[j]]/maxdaupt, dau_etas[dausort[j]]/maxdaudeta, dau_phis[dausort[j]]/maxdaudphi, dau_charges[dausort[j]]/maxdaucharge, dau_es[dausort[j]]/maxdaue, dau_mus[dausort[j]]/maxdaum, dau_rs[dausort[j]]/maxdaup, dau_chads[dausort[j]]/maxdauc, dau_nhads[dausort[j]]/maxdaun] if len(dau_pts)>j else [0.,0.,0.,0.,0.,0.,0.,0.,0.] for j in range(self.channel)]
            #seqs.append([[dau_pts[dausort[j]]/maxdaupt, dau_etas[dausort[j]]/maxdaudeta, dau_phis[dausort[j]]/maxdaudphi, dau_charges[dausort[j]]/maxdaucharge, dau_es[dausort[j]]/maxdaue, dau_mus[dausort[j]]/maxdaum, dau_rs[dausort[j]]/maxdaup, dau_chads[dausort[j]]/maxdauc, dau_nhads[dausort[j]]/maxdaun] if len(dau_pts)>j else [0.,0.,0.,0.,0.,0.,0.,0.,0.] for j in range(self.channel)])
          seqset.append(seqs)
        count+=1
        #jetset +=...
      else:
        #print(self.b,len(pts),len(etas),len(pids))
        count3+=1
      count2+=1
      ##label q=1 g=0
      self.b+=numjet
      """
      qptset.append(self.qjet.pt)
      qetaset.append(self.qjet.eta)
      qpidset.append(self.qjet.parton_id)
      if("c" in self.rc):
        maxchadpt=1.*sum(self.qjet.image_chad_pt_33)/self.normb
        maxnhadpt=1.*sum(self.qjet.image_nhad_pt_33)/self.normb
        maxelecpt=1.*sum(self.qjet.image_electron_pt_33)/self.normb
        maxmuonpt=1.*sum(self.qjet.image_muon_pt_33)/self.normb
        maxphotonpt=1.*sum(self.qjet.image_photon_pt_33)/self.normb
        maxchadmult=1.*sum(self.qjet.image_chad_mult_33)/self.normb
        maxnhadmult=1.*sum(self.qjet.image_nhad_mult_33)/self.normb
        maxelecmult=1.*sum(self.qjet.image_electron_mult_33)/self.normb
        maxmuonmult=1.*sum(self.qjet.image_muon_mult_33)/self.normb
        maxphotonmult=1.*sum(self.qjet.image_photon_mult_33)/self.normb
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
        if(self.order):
          maxdaupt=1.*sum(dau_pt)/self.normb
          maxdaudeta=1.*sum(dau_deta)/self.normb
          maxdaudphi=1.*sum(dau_dphi)/self.normb
          maxdaucharge=1.*sum(dau_charge)/self.normb
          maxdauc=1.*sum(dau_is_chad)/self.normb
          maxdaun=1.*sum(dau_is_nhad)/self.normb
          maxdaue=1.*sum(dau_is_e)/self.normb
          maxdaum=1.*sum(dau_is_mu)/self.normb
          maxdaup=1.*sum(dau_is_r)/self.normb
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
          """
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
    self.njset=np.array(njset)
    del njset
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
