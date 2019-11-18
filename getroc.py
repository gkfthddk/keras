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
for bpt in [200,1000]:
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
  #names=["asubdt{}pteta".format(pt),"asuzjrnn{}pteta21".format(pt),"asuzjcnn{}pt".format(pt)]
  #names=["asuzqcnn{}ptonly".format(pt),"asuqqcnn{}ptonly".format(pt)]
  #nl=['asuzqcnn{}ptetaptcut','asuqqcnn{}ptetaptcut','asuzjcnn{}ptptcut','asuzjcnn{}ptptcut']
  #names=["asubdt{}ptonly".format(pt),"asuzjrnn{}ptonly21".format(pt),"asuzjcnn{}ptonly3".format(pt)]
  names=["asubdt{}ptonly".format(pt),"asuzjcnn{}ptonly3".format(pt),"asuzjrnn{}ptonly21".format(pt)]
  plt.figure(figsize=(12, 8))
  plt.xlabel("Quark Jet Efficiency", fontsize=fs*1.2)
  plt.ylabel("Gluon Jet Rejection", fontsize=fs*1.2)
  plt.tick_params(labelsize=fs)
  for k in [1,0]:
    for savename in names:
      if("bdt" in savename):f=rt.TFile("xgb/"+savename+"get.root",'read')
      else:f=rt.TFile("save/asu/"+savename+"/get.root",'read')
      dq=f.Get("dq")
      zq=f.Get("zq")
      dg=f.Get("dg")
      zg=f.Get("zg")
      tree=[dq,zq,dg,zg]
      lab=["dijet","Z+jet"]
      mv=0
      if("zq" in savename and k==0):continue
      if("qq" in savename and k==1):continue
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
      print(savename,roc_auc_score(y,p))
      tnr=1-fpr
      label=""
      if("bdt" in savename):
        label="BDT"
        col="C2"
        mk="o"
        ms=1
        alpha=0.5
      if("cnn" in savename):
        label="CNN"
        col="C0"
        mk="D"
        ms=0.75
        alpha=0.8
      if("rnn" in savename):
        label="RNN"
        col="C1"
        mk="^"
        ms=1
        alpha=0.8
      if(k==0):
        label+="-dijet"
        #label+="-dijet ($p_T$ {}~{} GeV)".format(pt,int(pt*1.1))
        ls="--"
        fils='none'
        plt.plot(tpr,tnr,lw=4,alpha=alpha,label=label,linestyle=ls,color=col,)
        #plt.plot(tpr,tnr,lw=3.5,label=label,fillstyle=fils,linestyle=ls,marker=mk,markevery=15000,markersize=ms*fs*0.5,color=col,)
      if(k==1):
        #label+="-Z+jet ($p_T$ {}~{} GeV)".format(pt,int(pt*1.1))
        label+="-Z+jet"
        ls="-"
        fils='full'
        plt.plot(tpr,tnr,lw=3.5,alpha=alpha,label=label,linestyle=ls,color=col,)
        #plt.plot(tpr,tnr,lw=3.5,alpha=0.7,label=label,fillstyle=fils,linestyle=ls,marker=mk,markevery=15000,markersize=ms*fs*0.5,color=col,)
    f.Close()
  plt.title("jet $p_T$ range {}~{} GeV".format(pt,int(pt*1.1)),fontsize=fs)
  plt.legend(loc=3, fontsize=fs*0.9)
  plt.grid(alpha=0.6)
  plt.axis((0,1,0,1))
  plt.savefig("plots/asumixrocpt{}.pdf".format(pt),bbox_inches='tight',pad_inches=0.5,dpi=300)
  plt.savefig("plots/asumixrocpt{}.png".format(pt),bbox_inches='tight',pad_inches=0.5,dpi=300)
#plt.show()

#except:
#  pass

