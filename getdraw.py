import ROOT as rt
import matplotlib.pyplot as plt
import numpy as np
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--save",type=str,default="",help='rch')
parser.add_argument("--pt",type=int,default=500,help='pt range pt~pt*1.1')
args=parser.parse_args()
pt=args.pt

res=50
f=rt.TFile("save/asuzjcnn{}pt1/get.root".format(pt),'read')
dq=f.Get("dq")
zq=f.Get("zq")
dg=f.Get("dg")
zg=f.Get("zg")
tree=[dq,zq,dg,zg]
hmin=-2.4
hmax=2.4
var='eta'
h1=rt.TH1F("h1","", res, hmin,hmax)
dqhist=rt.TH1F("dqhist","dijet", res, hmin,hmax)
zqhist=rt.TH1F("zqhist","Z+jet", res, hmin,hmax)
dghist=rt.TH1F("dghist","dijet", res, hmin,hmax)
zghist=rt.TH1F("zghist","Z+jet", res, hmin,hmax)
hist=[dqhist,zqhist,dghist,zghist]
hi=['dqhist','zqhist','dghist','zghist']
label=['dijet-quark','zjet-quark','dijet-gluon','zjet-gluon']
mv=0
for i in range(4):
  tree[i].Draw("{}>>{}".format(var,hi[i]),"p>.5")
  hist[i].Scale(1.0/hist[i].Integral())
  if(mv<hist[i].GetMaximum()):mv=hist[i].GetMaximum()

ps=[[],[],[],[]]
for i in range(res):
  for j in range(4):
    ps[j].append(hist[j].GetBinContent(i+1))

for i in range(4):
  tree[i].Draw("{}>>{}".format(var,hi[i]),"p<.5")
  hist[i].Scale(1.0/hist[i].Integral())
  if(mv<hist[i].GetMaximum()):mv=hist[i].GetMaximum()

pscut=[[],[],[],[]]
for i in range(res):
  for j in range(4):
    pscut[j].append(hist[j].GetBinContent(i+1))

for i in range(0,2):
  xax=np.append(np.arange(-2.4,2.4,4.8/res)[:res],2.4)
  plt.plot(xax,np.append(ps[i],0),alpha=0.5,drawstyle='steps',label=label[i]+">.5")
  plt.plot(xax,np.append(pscut[i],0),alpha=0.5,drawstyle='steps',label=label[i]+"<.5")
plt.legend()
plt.show()
