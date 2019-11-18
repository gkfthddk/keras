import ROOT as rt
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--var",type=str,default="eta",help='')
parser.add_argument("--min",type=float,default=-2.4,help='')
parser.add_argument("--max",type=float,default=2.4,help='')
parser.add_argument("--pt",type=int,default=500,help='')

args=parser.parse_args()
pt=args.pt
var=args.var
res=50
hmin=0
hmax=1000
f=rt.TFile("save/asuzjcnn{}pt2/get.root".format(pt),'read')
dq=f.Get("dq")
zq=f.Get("zq")
dg=f.Get("dg")
zg=f.Get("zg")
tree=[dq,zq,dg,zg]
lab=["dijet","Z+jet"]
mv=0
#h1=rt.TH2F("h1","dijet-q",100,hmin,hmax,100,0,1)
#h2=rt.TH2F("h2","z+jet-q",100,hmin,hmax,100,0,1)
#h3=rt.TH2F("h3","dijet-g",100,hmin,hmax,100,0,1)
#h4=rt.TH2F("h4","z+jet-g",100,hmin,hmax,100,0,1)
h1=rt.TH1F("h1","dijet-q",100,0,1)
h2=rt.TH1F("h2","z+jet-q",100,0,1)
h3=rt.TH1F("h3","dijet-g",100,0,1)
h4=rt.TH1F("h4","z+jet-g",100,0,1)
hist=[h1,h2,h3,h4]
for k in range(2):
  y=[]
  p=[]
  for i in range(2):
    for j in range(tree[i*2+k].GetEntries()):
      tree[i*2+k].GetEntry(j)
      if(var=="pt"):
        if(abs(tree[i*2+k].pt)>args.max or abs(tree[i*2+k].pt)<args.min):continue
      if(var=="eta"):
        if(abs(tree[i*2+k].eta)>args.max or abs(tree[i*2+k].eta)<args.min):continue
      hist[i*2+k].Fill(tree[i*2+k].p)
      #hist[i*2+k].Fill(tree[i*2+k].pt,tree[i*2+k].p)
      #p.append(tree[i*2+k].p)
      #if(i<1):y.append(1)
      #else:y.append(0)

  #fpr,tpr,thresholds=roc_curve(y,p)
  #print(lab[k],roc_auc_score(y,p))
cav=rt.TCanvas("c","c",1000,1000)
#cav.Divide(2,2,0,0)
hist[0].Draw()
for i in range(1,4):
  #cav.cd(i+1)
  hist[i].SetLineColor(i+1)
  hist[i].Draw("Same")
