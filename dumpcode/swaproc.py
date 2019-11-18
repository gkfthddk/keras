import ROOT as rt
#python swaproc.py --save nocrasubdt1000 --pt 1000 --get nocut
#python swaproc.py --save mg5nocrasubdt1000 --pt 1000 --get nocut
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
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
    for k in [0,1]:
      y=[]
      p=[]
      for j in range(tree[k].GetEntries()):
        tree[k].GetEntry(j)
        if(abs(tree[k].pt)>args.pt*args.ptmax or abs(tree[k].pt)<args.pt*args.ptmin):continue
        if(abs(tree[k].eta)>args.eta+args.etabin or abs(tree[k].eta)<args.eta):continue
        if(args.gaus):
          if(random.random()>lgaus[k][int(250*tree[k].pt/args.pt)]):continue
        p.append(tree[k].p)
        y.append(1)
      for i in [2,3]:
        for j in range(tree[i].GetEntries()):
          tree[i].GetEntry(j)
          if(abs(tree[i].pt)>args.pt*args.ptmax or abs(tree[i].pt)<args.pt*args.ptmin):continue
          if(abs(tree[i].eta)>args.eta+args.etabin or abs(tree[i].eta)<args.eta):continue
          if(args.gaus):
            if(random.random()>lgaus[k][int(250*tree[i].pt/args.pt)]):continue
          p.append(tree[i].p)
          y.append(0)
        fpr,tpr,thresholds=roc_curve(y,p)
        tnr=1-fpr
        diff=1.
        diffi=0
        for l in range(len(tpr)):
          if(abs(tpr[l]-0.5)<diff):
            diff=abs(tpr[l]-0.5)
            diffi=l
        print(args.savename,"q "+lab[k],"g "+lab[i-2],round(roc_auc_score(y,p),5),round(fpr[diffi],5),diffi)
    f.Close()
  #except:
  #  pass

