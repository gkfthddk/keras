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

if(args.isz==0):iii=1
if(args.isz==1):iii=2
if(args.isz==-1):iii=3
rc=""
onehot=0
if(args.isz==0):
  tqdata="Data/zj_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  tgdata="Data/jj_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  train=wkiter([tqdata,tgdata],batch_size=batch_size,end=args.end*0.7,istrain=1,rc=rc,onehot=onehot,etabin=args.etabin,pt=args.pt,ptmin=args.ptmin,ptmax=args.ptmax)
  valid1=wkiter([vzjdata,vjjdata],batch_size=batch_size,begin=0.8*args.end,end=args.end*1.,rc=rc,onehot=onehot,etabin=args.etabin,pt=args.pt,ptmin=args.ptmin,ptmax=args.ptmax)
test2=wkiter([vzqdata,vzgdata],batch_size=batch_size,begin=args.end*0.2,end=args.end*0.6,rc=rc,onehot=onehot,channel=args.channel,order=args.order,eta=0,etabin=2.4)
test3=wkiter([vqqdata,vggdata],batch_size=batch_size,begin=args.end*0.2,end=args.end*0.6,rc=rc,onehot=onehot,channel=args.channel,order=args.order,eta=0,etabin=2.4)
entries=test2.totalnum()
print ("test   ",entries)
print(args.pt)
gen=train.next()
genv=valid1.next()
#epoch=eval(open(savename+"/history").readline())+1
X,Y=next(gen)
Y=np.array(Y)[:,0]
X=np.array(X[0])
xv,yv=next(genv)
xv=np.array(xv[0])
yv=np.array(yv[:,0])
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

py=[]
py.append(model.predict_proba(test3.qjetset)[:,1])
py.append(model.predict_proba(test2.qjetset)[:,1])
py.append(model.predict_proba(test3.gjetset)[:,1])
py.append(model.predict_proba(test2.gjetset)[:,1])
jetset=[[],[],[],[]]
jetset[0].append(test3.qptset)
jetset[0].append(test3.qetaset)
jetset[1].append(test2.qptset)
jetset[1].append(test2.qetaset)
jetset[2].append(test3.gptset)
jetset[2].append(test3.getaset)
jetset[3].append(test2.gptset)
jetset[3].append(test2.getaset)
for i in range(5):
  jetset[0].append(test3.qjetset[:,i])
  jetset[1].append(test2.qjetset[:,i])
  jetset[2].append(test3.gjetset[:,i])
  jetset[3].append(test2.gjetset[:,i])

varss=['pt','eta','ptd','major_axis','minor_axis','chad_mult','nhad_mult']
#varss=['pt','eta','ptd']
cutter=""
for i in range(len(varss)):
  canv=rt.TCanvas("canv","canv",1000,1000)
  canv.Divide(2,2)
  ma=0;
  mi=0;
  for j in range(4):
    if(ma<max(jetset[j][i])):ma=max(jetset[j][i])
    if(mi>min(jetset[j][i])):mi=min(jetset[j][i])
  res=100
  hists=[]
  if("mult" in varss[i]):
    hists.append(rt.TH2F("qqhist{}".format(varss[i]),"qq {}".format(varss[i]),int(ma),0,int(ma),res,0,1))
    hists.append(rt.TH2F("zqhist{}".format(varss[i]),"zq {}".format(varss[i]),int(ma),0,int(ma),res,0,1))
    hists.append(rt.TH2F("gghist{}".format(varss[i]),"gg {}".format(varss[i]),int(ma),0,int(ma),res,0,1))
    hists.append(rt.TH2F("zghist{}".format(varss[i]),"zg {}".format(varss[i]),int(ma),0,int(ma),res,0,1))
  elif("eta" in varss[i]):
    hists.append(rt.TH2F("qqhist{}".format(varss[i]),"qq {}".format(varss[i]),100,-2.4,2.4,res,0,1))
    hists.append(rt.TH2F("zqhist{}".format(varss[i]),"zq {}".format(varss[i]),100,-2.4,2.4,res,0,1))
    hists.append(rt.TH2F("gghist{}".format(varss[i]),"gg {}".format(varss[i]),100,-2.4,2.4,res,0,1))
    hists.append(rt.TH2F("zghist{}".format(varss[i]),"zg {}".format(varss[i]),100,-2.4,2.4,res,0,1))
  else:
    hists.append(rt.TH2F("qqhist{}".format(varss[i]),"qq {}".format(varss[i]),100,mi,ma,res,0,1))
    hists.append(rt.TH2F("zqhist{}".format(varss[i]),"zq {}".format(varss[i]),100,mi,ma,res,0,1))
    hists.append(rt.TH2F("gghist{}".format(varss[i]),"gg {}".format(varss[i]),100,mi,ma,res,0,1))
    hists.append(rt.TH2F("zghist{}".format(varss[i]),"zg {}".format(varss[i]),100,mi,ma,res,0,1))


  for j in range(4):
    for k in range(len(py[j])):
      hists[j].Fill(jetset[j][i][k],py[j][k])
    hists[j].Scale(1./len(py[j]))
    canv.cd(j+1)
    hists[j].Draw('colz')
  canv.SaveAs("plots/bdtvspred{}_{}.png".format(args.pt,varss[i]))
  canv.Clear()
