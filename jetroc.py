import ROOT as rt
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import argparse
import os
import random
fs=25
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
for bpt in [200,500]:
  args.pt=bpt
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
  pt=args.pt
  res=100
  #os.system("ls save/"+args.savename+"/get.root")
  if(args.gaus):
    jgaus=eval(open("jjgaus.txt").readline())
    zgaus=eval(open("zjgaus.txt").readline())
    lgaus=[jgaus,zgaus]
    #try:
  names=["dualmrdr5{}non".format(pt),"genrdr5{}non".format(pt),"bdt1pt{}1".format(pt),"genbdt1pt{}1".format(pt)]
  #names=["dualc5{}non".format(pt),"dualc5{}con".format(pt),"dualmrdr5{}non".format(pt),"dualmrdr5{}con".format(pt),"bdt1pt{}1".format(pt),"bdt1pt{}3".format(pt)]
  plt.figure(figsize=(12, 8))
  plt.xlabel("Quark Jet Efficiency", fontsize=fs*1.2)
  plt.ylabel("Gluon Jet Rejection", fontsize=fs*1.2)
  plt.tick_params(labelsize=fs)
  for savename in names:
    if("bdt" in savename):
      f1=rt.TFile("xgb/"+savename+"get.root",'read')
      f2=rt.TFile("xgb/"+savename[:savename.find("bdt")+3]+"2"+savename[savename.find("bdt")+4:]+"get.root",'read')
      q1=f1.Get("dg")
      q2=f2.Get("dg")
      g1=f1.Get("dq")
      g2=f2.Get("dq")
    else:
      f=rt.TFile("save/"+savename+"/get.root",'read')
      print("save/"+savename+"/get.root")
      q1=f.Get("q1")
      q2=f.Get("q2")
      g1=f.Get("g1")
      g2=f.Get("g2")
    tree=[q1,q2,g1,g2]
    lab=["first","second"]
    y=[]
    p=[]
    for k in [1,0]:
      mv=0
      if("zq" in savename and k==0):continue
      if("qq" in savename and k==1):continue
      for i in range(2):
        for j in range(tree[i*2+k].GetEntries()):
          tree[i*2+k].GetEntry(j)
          #if(abs(tree[i*2+k].pt)>args.pt*args.ptmax or abs(tree[i*2+k].pt)<args.pt*args.ptmin):continue
          if(abs(tree[i*2+k].eta)>args.eta+args.etabin or abs(tree[i*2+k].eta)<args.eta):continue
          if(args.gaus):
            if(random.random()>lgaus[k][int(250*tree[i*2+k].pt/args.pt)]):continue
          if(i<1):
            y.append(1)
            if("bdt" in savename):
              p.append(tree[i*2+k].p)
            else:
              p.append(tree[i*2+k].p)
          else:
            y.append(0)
            if("bdt" in savename):
              p.append(tree[i*2+k].p)
            else:
              p.append(1-tree[i*2+k].p)
    fpr,tpr,thresholds=roc_curve(y,p)
    print(savename,roc_auc_score(y,p))
    tnr=1-fpr
    label=""
    if("bdt" in savename):
      label="BDT"
      col="C2"
      mk="o"
      ms=1
      alpha=0.5
    elif("dr" in savename):
      label="RNN"
      col="C0"
      mk="D"
      ms=0.75
      alpha=0.8
    elif("lc" in savename):
      label="CNN"
      col="C1"
      mk="^"
      ms=1
      alpha=0.8
    else:
      label="MIN"
      col="C1"
      mk="^"
      ms=1
      alpha=0.8
    if("non" in savename or savename[-1]=="1"):
      label+=" - AUC:{}".format(round(roc_auc_score(y,p),4))
      #label+="-dijet ($p_T$ {}~{} GeV)".format(pt,int(pt*1.1))
      ls="--"
      fils='none'
      plt.plot(tpr,tnr,lw=4,alpha=alpha,label=label,linestyle=ls,color=col,)
      #plt.plot(tpr,tnr,lw=3.5,label=label,fillstyle=fils,linestyle=ls,marker=mk,markevery=15000,markersize=ms*fs*0.5,color=col,)
    else:
      #label+="-Z+jet ($p_T$ {}~{} GeV)".format(pt,int(pt*1.1))
      label+="*- AUC:{}".format(round(roc_auc_score(y,p),4))
      ls="-"
      fils='full'
      plt.plot(tpr,tnr,lw=3.5,alpha=alpha,label=label,linestyle=ls,color=col,)
      #plt.plot(tpr,tnr,lw=3.5,alpha=0.7,label=label,fillstyle=fils,linestyle=ls,marker=mk,markevery=15000,markersize=ms*fs*0.5,color=col,)
    try:
      f.Close()
    except:
      f1.Close()
      f2.Close()
  plt.title("jet $p_T$ range {}~{} GeV".format(pt,int(pt*1.1)),fontsize=fs)
  plt.legend(loc=3, fontsize=fs*0.9)
  plt.grid(alpha=0.6)
  plt.axis((0,1,0,1))
  #plt.savefig("plots/dualrocpt{}.pdf".format(pt),bbox_inches='tight',pad_inches=0.5,dpi=300)
  plt.savefig("plots/genrocpt{}.png".format(pt),bbox_inches='tight',pad_inches=0.5,dpi=300)
  #plt.savefig("plots/dualrocpt{}.png".format(pt),bbox_inches='tight',pad_inches=0.5,dpi=300)
#plt.show()

#except:
#  pass

