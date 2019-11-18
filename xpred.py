import xgboost as xgb
from sklearn.metrics import roc_auc_score, auc, roc_curve
import pandas as pd
import numpy as np
import pickle
from xiter import *
import argparse
import matplotlib.pyplot as plt

data=pd.read_csv('xgb/random-search.csv')

parser=argparse.ArgumentParser()
parser.add_argument("--end",type=float,default=100000.,help='end ratio')
parser.add_argument("--save",type=str,default="random-search.csv",help='save name')
parser.add_argument("--right",type=str,default="/scratch/yjdata/gluon100_img",help='which train sample (qq,gg,zq,zg)')
parser.add_argument("--pt",type=int,default=200,help='pt range pt~pt*1.1')
parser.add_argument("--ptmin",type=float,default=0.,help='pt range pt~pt*1.1')
parser.add_argument("--ptmax",type=float,default=2.,help='pt range pt~pt*1.1')
parser.add_argument("--batch_size",type=int,default=100000,help='batch_size')
parser.add_argument("--gpu",type=int,default=0,help='gpu number')
parser.add_argument("--isz",type=int,default=0,help='0 or z or not')
parser.add_argument("--channel",type=int,default=30,help='sequence channel')
parser.add_argument("--order",type=int,default=1,help='pt ordering')
parser.add_argument("--eta",type=float,default=0.,help='end ratio')
parser.add_argument("--etabin",type=float,default=1.,help='end ratio')

args=parser.parse_args()

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
batch_size=args.batch_size
vzjdata="Data/zj_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
vjjdata="Data/jj_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
vzqdata="Data/zq_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
vzgdata="Data/zg_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
vqqdata="Data/qq_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
vggdata="Data/gg_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))

rc=""
onehot=0
if(args.isz==0):
  tqdata="Data/zj_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  tgdata="Data/jj_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  train=wkiter([tqdata,tgdata],batch_size=batch_size,end=args.end*0.7,istrain=1,rc=rc,onehot=onehot,etabin=args.etabin,pt=args.pt,ptmin=args.ptmin,ptmax=args.ptmax)
  valid1=wkiter([vzjdata,vjjdata],batch_size=batch_size,begin=0.8*args.end,end=args.end*1.,rc=rc,onehot=onehot,etabin=args.etabin,pt=args.pt,ptmin=args.ptmin,ptmax=args.ptmax)
elif(args.isz==1):
  tqdata="Data/zq_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  tgdata="Data/zg_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  train=wkiter([tqdata,tgdata],batch_size=batch_size,begin=0.6*args.end,end=args.end*1.,istrain=1,rc=rc,onehot=onehot,etabin=args.etabin,pt=args.pt,ptmin=args.ptmin,ptmax=args.ptmax)
else:
  tqdata="Data/qq_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  tgdata="Data/gg_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  train=wkiter([tqdata,tgdata],batch_size=batch_size,begin=0.6*args.end,end=args.end*1.,istrain=1,rc=rc,onehot=onehot,etabin=args.etabin,pt=args.pt,ptmin=args.ptmin,ptmax=args.ptmax)
test2=wkiter([vzqdata,vzgdata],batch_size=batch_size,begin=args.end*0.,end=args.end*0.6,rc=rc,onehot=onehot,channel=args.channel,order=args.order,eta=0,etabin=2.4)
test3=wkiter([vqqdata,vggdata],batch_size=batch_size,begin=args.end*0.,end=args.end*0.6,rc=rc,onehot=onehot,channel=args.channel,order=args.order,eta=0,etabin=2.4)
entries=test2.totalnum()
print ("test   ",entries)
print(args.pt)
gen=train.next()
#genv=valid1.next()
#epoch=eval(open(savename+"/history").readline())+1
X,Y=next(gen)
Y=np.array(Y)[:,0]
X=np.array(X[0])
#xv,yv=next(genv)
#xv=np.array(xv[0])
#yv=np.array(yv[:,0])
test2.reset()
test3.reset()
#genz=test2.next()
#genq=test3.next()
#xz,yz=next(genz)
#xq,yq=next(genq)
#xz=np.array(xz[0])
#xq=np.array(xq[0])
#yz=np.array(yz[:,0])
#yq=np.array(yq[:,0])
csv=pd.read_csv("xgb/{}-{}.csv".format(args.save,args.pt))
#csv=pd.read_csv(args.save)
best=csv.loc[csv["mean_test_score"].idxmax()]
model=xgb.XGBClassifier(objective='binary:logistic',tree_method="gpu_exact",**best)
#model=pickle.load(open("xgb/bdt100pickle-{}.dat".format(args.pt)))
model.fit(X,Y,eval_metric="auc")

f=rt.TFile("xgb/{}get.root".format(args.save),"recreate")
dq=rt.TTree("dq","dq tree")
dg=rt.TTree("dg","dg tree")
zq=rt.TTree("zq","zq tree")
zg=rt.TTree("zg","zg tree")
p=array('f',[0.])
pt=array('f',[0.])
eta=array('f',[0.])
dq.Branch("p",p,"p/F")
dq.Branch("pt",pt,"pt/F")
dq.Branch("eta",eta,"eta/F")
dg.Branch("p",p,"p/F")
dg.Branch("pt",pt,"pt/F")
dg.Branch("eta",eta,"eta/F")
zq.Branch("p",p,"p/F")
zq.Branch("pt",pt,"pt/F")
zq.Branch("eta",eta,"eta/F")
zg.Branch("p",p,"p/F")
zg.Branch("pt",pt,"pt/F")
zg.Branch("eta",eta,"eta/F")



"""py=model.predict_proba(xv)[:,1]
t_fpr,t_tpr,thresholds=roc_curve(yv,py)
t_tnr=1-t_fpr
#print(args.pt,"v",auc(t_fpr,t_tpr))
va=auc(t_fpr,t_tpr)
plt.figure(1)

q=[]
g=[]
for i in range(len(py)):
  if(yv[i]==1):q.append(py[i])
  else:g.append(py[i])
plt.hist(q,bins=50,weights=np.ones_like(q),histtype='step',alpha=0.5,label='vq')
plt.hist(g,bins=50,weights=np.ones_like(g),histtype='step',alpha=0.5,label='vg')
"""
#xz,yz=next(genz)
#xq,yq=next(genq)
bpy=model.predict_proba(test2.gjetset)[:,1]
bpt=test2.gptset
beta=test2.getaset
for i in range(len(bpy)):
  p[0]=bpy[i]
  pt[0]=bpt[i]
  eta[0]=beta[i]
  zg.Fill()
bpy=model.predict_proba(test2.qjetset)[:,1]
bpt=test2.qptset
beta=test2.qetaset
for i in range(len(bpy)):
  p[0]=bpy[i]
  pt[0]=bpt[i]
  eta[0]=beta[i]
  zq.Fill()
bpy=model.predict_proba(test3.gjetset)[:,1]
bpt=test3.gptset
beta=test3.getaset
for i in range(len(bpy)):
  p[0]=bpy[i]
  pt[0]=bpt[i]
  eta[0]=beta[i]
  dg.Fill()
bpy=model.predict_proba(test3.qjetset)[:,1]
bpt=test3.qptset
beta=test3.qetaset
for i in range(len(bpy)):
  p[0]=bpy[i]
  pt[0]=bpt[i]
  eta[0]=beta[i]
  dq.Fill()
f.Write()
f.Close()
#t_fpr,t_tpr,thresholds=roc_curve(yz,py)
#t_tnr=1-t_fpr
#print(args.pt,"z",auc(t_fpr,t_tpr))
#za=auc(t_fpr,t_tpr)
q=[]
g=[]
#for i in range(len(py)):
#  if(yz[i]==1):q.append(py[i])
#  else:g.append(py[i])
#plt.hist(q,bins=50,weights=np.ones_like(q),histtype='step',alpha=0.5,label='zq')
#plt.hist(g,bins=50,weights=np.ones_like(g),histtype='step',alpha=0.5,label='zg')

#t_fpr,t_tpr,thresholds=roc_curve(yq,py)
#t_tnr=1-t_fpr
#print(args.pt,"q",auc(t_fpr,t_tpr),len(yq))
#qa=auc(t_fpr,t_tpr)
q=[]
g=[]
#for i in range(len(py)):
#  if(yq[i]==1):q.append(py[i])
#  else:g.append(py[i])
#plt.hist(q,bins=50,weights=np.ones_like(q),histtype='step',alpha=0.5,label='qq')
#plt.hist(g,bins=50,weights=np.ones_like(g),histtype='step',alpha=0.5,label='qg')
#print(args.pt,va,qa,za)
#plt.legend()
#plt.savefig("xgb/bdtout{}.png".format(args.pt))
