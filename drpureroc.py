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
parser.add_argument("--pt",type=int,default=100,help='')
parser.add_argument("--get",type=str,default="pt",help='')
parser.add_argument("--gaus",type=int,default=0,help='')
args=parser.parse_args()
for bpt in [100]:
  args.pt=bpt
  if("pt" in args.get):
    if(args.pt==100):
      args.ptmin=0.815
      args.ptmax=1.159
  pt=args.pt
  res=100
  #os.system("ls save/"+args.savename+"/get.root")
  #names=["drbox/qg423rot50out.npz","asuqqcnn{}ptonly".format(pt)]
  #names=["drbox/ep423rot50out.npz","drbox/gp423rot50out.npz"]
  #names=["drbox/qg423b64img50out.npz","drbox/qg2or501024out.npz","drbox/qg2onr502048out.npz"]
  #names=["drbox/qg423img50out.npz","drbox/qg423rot50out.npz"]
  names=["drqgpout.npz","drqgp0out.npz","drqgcout.npz","drqg0dout.npz","qgpntout.npz"]
  labels=["30pixel","90pixel","zoom","90d","pnt"]
  #names=["drbox/qg423b64img50out.npz","drbox/qg2onr502048out.npz"]
  plt.figure(figsize=(12, 8))
  if("qg" in names[0]):
    plt.xlabel("Quark Jet Efficiency", fontsize=fs*1.2)
    plt.ylabel("Gluon Jet Rejection", fontsize=fs*1.2)
  if("ep" in names[0]):
    plt.xlabel("Electron, Gamma Efficiency", fontsize=fs*1.1)
    plt.ylabel("Pion Rejection", fontsize=fs*1.2)
  plt.tick_params(labelsize=fs)
  for k in [0]:
    for num in range(len(names)):
      savename = names[num]
      y=[]
      p=[]
      if("npz" in savename):
        f=np.load("drbox/"+savename)
        if("or" in savename):
          try:
            y=f["testY"][:,1]
            p=f["bp"][:,1]
          except:
            y=f["y"][:,1]
            p=f["p"][:,1]
        else:
          try:
            y=f["testY"][:,0]
            p=f["bp"][:,0]
          except:
            y=f["y"][:,0]
            p=f["p"][:,0]
        f.close()
      else:
        f=rt.TFile("save/asu/"+savename+"/get.root",'read')
        dq=f.Get("dq")
        dg=f.Get("dg")
        for sample in ["dq","dg"]:
          tree=f.Get(sample)
          for j in range(tree.GetEntries()):
            tree.GetEntry(j)
            if(abs(tree.pt)>args.pt*args.ptmax or abs(tree.pt)<args.pt*args.ptmin):continue
            p.append(tree.p)
            if(sample=="dq"):y.append(1)
            else:y.append(0)
        f.Close()
      fpr,tpr,thresholds=roc_curve(y,p)
      print(savename,roc_auc_score(y,p))
      tnr=1-fpr
      label=""
      if("npz" in savename):
        mk="D"
        #mk="^"
        ms=0.75
        alpha=0.8
        ls="-"
        label=labels[num]
      fils='none'
      label+=" - AUC:{}".format(round(roc_auc_score(y,p),4))
      plt.plot(tpr,tnr,lw=4,alpha=alpha,label=label,linestyle=ls,)
      print(1)
  plt.legend(loc=3, fontsize=fs*0.9)
  plt.grid(alpha=0.6)
  plt.axis((0,1,0,1))
  if("ep" in names[0]):
    plt.axis((0.9,1,0.9,1))
  #plt.savefig("plots/asupurerocpt{}.png".format(pt),bbox_inches='tight',pad_inches=0.5,dpi=300)
plt.show()

#except:
#  pass

