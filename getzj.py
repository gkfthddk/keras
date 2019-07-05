import xgboost as xgb
from sklearn.metrics import roc_auc_score, auc, roc_curve
import pandas as pd
import numpy as np
import pickle
from xiter import *
import argparse
import matplotlib.pyplot as plt


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
parser.add_argument("--etabin",type=float,default=2.4,help='end ratio')

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
#if(args.isz==0):
tqdata="Data/zj_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
tgdata="Data/jj_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
  #train=wkiter([tqdata,tgdata],batch_size=batch_size,end=args.end*0.7,istrain=1,rc=rc,onehot=onehot,etabin=args.etabin,pt=args.pt,ptmin=args.ptmin,ptmax=args.ptmax)
test2=wkiter([vzqdata,vzgdata],batch_size=batch_size,begin=args.end*0.2,end=args.end*0.6,rc=rc,onehot=onehot,channel=args.channel,order=args.order,eta=0,etabin=2.4)
test3=wkiter([vqqdata,vggdata],batch_size=batch_size,begin=args.end*0.2,end=args.end*0.6,rc=rc,onehot=onehot,channel=args.channel,order=args.order,eta=0,etabin=2.4)
entries=test2.totalnum()
print ("test   ",entries)
print(args.pt)
#gen=train.next()
#X,Y=next(gen)
#Y=np.array(Y)[:,0]
#X=np.array(X[0])
test2.reset()
test3.reset()
savename="save/"+str(args.save)
if(args.isz==1):
  f1=rt.TFile("{}/get.root".format(savename.format("zq")),"read")
  f2=rt.TFile("{}/get.root".format(savename.format("qq")),"read")
  dq=f2.Get("dq")
  dg=f2.Get("dg")
  zq=f1.Get("zq")
  zg=f1.Get("zg")
else:
  f=rt.TFile("{}/get.root".format(savename),"read")
  dq=f.Get("dq")
  dg=f.Get("dg")
  zq=f.Get("zq")
  zg=f.Get("zg")
tree=[dq,zq,dg,zg]
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
    count=0
    for k in range(tree[j].GetEntries()):
      tree[j].GetEntry(k)
      if(tree[j].pt!=jetset[j][0][k+count]):
        count+=1
      if(tree[j].pt!=jetset[j][0][k+count]):
        count+=1
      hists[j].Fill(jetset[j][i][k+count],tree[j].p)
    hists[j].Scale(1./(tree[j].GetEntries()))
    canv.cd(j+1)
    hists[j].Draw('colz')
  canv.SaveAs("plots/{}pred{}_{}.png".format(args.save,args.pt,varss[i]))
  canv.Clear()
