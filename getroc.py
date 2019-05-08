import ROOT as rt
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--var",type=str,default="eta",help='')
parser.add_argument("--min",type=float,default=-2.4,help='')
parser.add_argument("--max",type=float,default=2.4,help='')
parser.add_argument("--pt",type=int,default=1000,help='')
parser.add_argument("--onpt",type=int,default=0,help='')

args=parser.parse_args()
pt=args.pt
res=50
if(args.var=="pt"):
  if(args.min<10):
    args.min=args.min*args.pt
  if(args.max<10):
    args.max=args.max*args.pt

for onpt in ["nopt","onpt"]:
  try:
    if(onpt=="onpt"):
      f=rt.TFile("save/asuzjcnn{}pt/get.root".format(pt),'read')
    else:
      f=rt.TFile("save/asuzjcnn{}/get.root".format(pt),'read')
    dq=f.Get("dq")
    zq=f.Get("zq")
    dg=f.Get("dg")
    zg=f.Get("zg")
    tree=[dq,zq,dg,zg]
    lab=["dijet","Z+jet"]
    mv=0
    for k in range(2):
      y=[]
      p=[]
      for i in range(2):
        for j in range(tree[i*2+k].GetEntries()):
          tree[i*2+k].GetEntry(j)
          if(args.var=="pt"):
            if(abs(tree[i*2+k].pt)>args.max or abs(tree[i*2+k].pt)<args.min):continue
          if(args.var=="eta"):
            if(abs(tree[i*2+k].eta)>args.max or abs(tree[i*2+k].eta)<args.min):continue
          p.append(tree[i*2+k].p)
          if(i<1):y.append(1)
          else:y.append(0)
      #fpr,tpr,thresholds=roc_curve(y,p)
      print(onpt,lab[k],roc_auc_score(y,p))
    f.Close()
  except:
    pass

