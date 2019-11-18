import ROOT as rt
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np
import argparse
import os
import random
parser=argparse.ArgumentParser()
parser.add_argument("--var",type=str,default="eta",help='')
parser.add_argument("--savename",type=str,default="savemae",help='')
parser.add_argument("--eta",type=float,default=0,help='')
parser.add_argument("--etabin",type=float,default=2.4,help='')
parser.add_argument("--ptmin",type=float,default=0,help='')
parser.add_argument("--ptmax",type=float,default=2.,help='')
parser.add_argument("--pt",type=int,default=1000,help='')
parser.add_argument("--get",type=str,default="",help='')
parser.add_argument("--gaus",type=int,default=0,help='')
args=parser.parse_args()
if("pt" in args.get):
  if(args.pt==100):
    args.ptmin=0.815
    args.ptmax=1.159
  if(args.pt==200):
    args.ptmin=0.819
    args.ptmax=1.123
  if(args.pt==500):
    args.ptmin=0.821
    args.ptmax=1.093
  if(args.pt==1000):
    args.ptmin=0.8235
    args.ptmax=1.076
if("eta" in args.get):
  args.etabin=1
if("acut" in args.get):
  if(args.pt==100):
    args.ptmin=0.815
    args.ptmax=1.159
  if(args.pt==200):
    args.ptmin=0.819
    args.ptmax=1.123
  if(args.pt==500):
    args.ptmin=0.821
    args.ptmax=1.093
  if(args.pt==1000):
    args.ptmin=0.8235
    args.ptmax=1.076
  args.etabin=1
pt=args.pt
res=100
#os.system("ls save/"+args.savename+"/get.root")
if(args.gaus):
  jgaus=eval(open("jjgaus.txt").readline())
  zgaus=eval(open("zjgaus.txt").readline())
  lgaus=[jgaus,zgaus]
for num in range(1,2):
  #try:
    if("bdt" in args.savename):f=rt.TFile("xgb/"+args.savename+"get.root",'read')
    else:f=rt.TFile("save/"+args.savename+"/get.root",'read')
    dq=f.Get("dq")
    zq=f.Get("zq")
    dg=f.Get("dg")
    zg=f.Get("zg")
    tree=[dq,zq,dg,zg]
    lab=["dijet","Z+jet"]
    mv=0
    for k in range(2):
      if("zq" in args.savename and k==0):continue
      if("qq" in args.savename and k==1):continue
      y=[]
      p=[]
      for i in range(2):
        for j in range(tree[i*2+k].GetEntries()):
          tree[i*2+k].GetEntry(j)
          if(abs(tree[i*2+k].pt)>args.pt*args.ptmax or abs(tree[i*2+k].pt)<args.pt*args.ptmin):continue
          if(abs(tree[i*2+k].eta)>args.eta+args.etabin or abs(tree[i*2+k].eta)<args.eta):continue
          if(args.gaus):
            if(random.random()>lgaus[k][int(250*tree[i*2+k].pt/args.pt)]):continue
          p.append(tree[i*2+k].p)
          if(i<1):y.append(1)
          else:y.append(0)
      fpr,tpr,thresholds=roc_curve(y,p)
      tnr=1-fpr
      diff=1.
      diffi=0
      for i in range(len(tpr)):
        if(abs(tpr[i]-0.5)<diff):
          diff=abs(tpr[i]-0.5)
          diffi=i
      plt.plot(tpr,tnr)
      plt.axis((0,1,0,1))
      plt.grid()
      plt.show()
      print(args.savename,lab[k],round(roc_auc_score(y,p),5),round(fpr[diffi],5),diffi)
    f.Close()
  #except:
  #  pass

