import ROOT as rt
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import argparse
import os
import random
import scipy.stats
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
parser.add_argument("--parton",type=int,default=0,help='')
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
#if("eta" in args.get):
#  args.etabin=1
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
if("npz" in args.savename):
  args.ptmin=1.
  args.ptmax=1.2
pt=args.pt
res=100
#os.system("ls save/"+args.savename+"/get.root")
if("bdt" in args.savename):f=rt.TFile("xgb/"+args.savename+"get.root",'read')
else:f=rt.TFile("save/"+args.savename+"/getd.root",'read')
tree={}
for i in range(2):
  tree["q{}".format(i+1)]=f.Get("q{}".format(i+1))
  tree["g{}".format(i+1)]=f.Get("g{}".format(i+1))
  mv=0
  y=[]
  p=[]
  for j in ["q{}","g{}"]:
    tr=j.format(i+1)
    for k in range(tree[tr].GetEntries()):
      tree[tr].GetEntry(k)
      if(abs(tree[tr].pt)>args.pt*args.ptmax or abs(tree[tr].pt)<args.pt*args.ptmin):continue
      if(abs(tree[tr].eta)>args.eta+args.etabin or abs(tree[tr].eta)<args.eta):continue
      if(j=="q{}"):
        y.append(1)
        p.append(tree[tr].p)
      if(j=="g{}"):
        y.append(0)
        p.append(1.-tree[tr].p)
  sem=scipy.stats.sem(p)
  fpr,tpr,thresholds=roc_curve(y,p)
  tnr=1-fpr
  diff=1.
  diffi=0
  for j in range(len(tpr)):
    if(abs(tpr[j]-0.5)<diff):
      diff=abs(tpr[j]-0.5)
      diffi=j
  print(args.savename,i+1,round(roc_auc_score(y,p),3),round(fpr[diffi],5),diffi)
f.Close()
#except:
#  pass

