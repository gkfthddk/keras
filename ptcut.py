import numpy as np
import datetime
import random
import ROOT as rt
from ROOT import gPad,gStyle
import math
import sys
from array import array
import matplotlib.pyplot as plt
pt=1000
dqf=rt.TFile("Data/qq_pt_{0}_{1}.root".format(pt,int(pt*1.1)),'read')
dq=dqf.Get("jetAnalyser")
dgf=rt.TFile("Data/gg_pt_{0}_{1}.root".format(pt,int(pt*1.1)),'read')
dg=dgf.Get("jetAnalyser")
zqf=rt.TFile("Data/zq_pt_{0}_{1}.root".format(pt,int(pt*1.1)),'read')
zq=zqf.Get("jetAnalyser")
zgf=rt.TFile("Data/zg_pt_{0}_{1}.root".format(pt,int(pt*1.1)),'read')
zg=zgf.Get("jetAnalyser")
arnum=16
lg=128
qpt=rt.TH1F("qpt","quark",lg,0,lg)
gpt=rt.TH1F("gpt","gluon",lg,0,lg)
dqe=dq.GetEntries()
dge=dg.GetEntries()
zqe=zq.GetEntries()
zge=zg.GetEntries()
lo=[0,0,0,0]
res=1.
for i in range(int(pt*0.7/res),int(pt*0.9/res))+range(int(pt*1./res),int(pt*1.3/res)):
  dqp=1.*dq.GetEntries("pt<{}".format(i*res))/dqe
  dgp=1.*dg.GetEntries("pt<{}".format(i*res))/dge
  zqp=1.*zq.GetEntries("pt<{}".format(i*res))/zqe
  zgp=1.*zg.GetEntries("pt<{}".format(i*res))/zge
  li=[dqp,dgp,zqp,zgp]
  for j in range(4):
    if(lo[j]==0):
      if(li[j]>=0.15):
        print(round(i*res/pt,5),li[j])
        lo[j]=1
    if(lo[j]==1):
      if(li[j]>=0.849):
        print(round(i*res/pt,5),li[j])
        lo[j]=2
  if(lo==[2,2,2,2]):
    break
"""
canv=rt.TCanvas("canv","Paticles' mean p_{T} by ordr in each jets",1000,700)
canv.SetLogy()
canv.cd(1)
gStyle.SetOptStat(0)
qpt.SetLineColor(4)
qpt.SetLineWidth(3)
qpt.GetXaxis().SetTitle("Order")
qpt.GetYaxis().SetTitle("Mean p_{T} GeV")
gpt.SetLineColor(2)
gpt.SetLineWidth(3)
gpt.GetXaxis().SetTitle("Order")
gpt.GetYaxis().SetTitle("Mean p_{T} GeV")
gpt.Scale(1.0 / gpt.Integral())
qpt.Scale(1.0 / qpt.Integral())
gpt.Draw("HIST")
qpt.Draw("HIST Same")
gPad.BuildLegend()
qpt.SetTitle("Mean p_{T} by odering at jet p_{T} "+"{0}~{1}".format(pt,int(pt*1.1)))
#plt.plot(range(20),gpt,range(20),qpt)
#plt.show()
qpt.Fit("expo")
gpt.Fit("expo")"""
dqf.Close()
dgf.Close()
zqf.Close()
zgf.Close()
