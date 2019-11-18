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
qf=rt.TFile("Data/zq_pt_{0}_{1}.root".format(pt,int(pt*1.1)),'read')
qjet=qf.Get("jetAnalyser")
gf=rt.TFile("Data/zg_pt_{0}_{1}.root".format(pt,int(pt*1.1)),'read')
gjet=gf.Get("jetAnalyser")
arnum=16
lg=128
qpt=rt.TH1F("qpt","quark",64,0,64)
gpt=rt.TH1F("gpt","gluon",64,0,64)
#qim = array('B', [0]*(3*(arnum*2+1)*(arnum*2+1)))
#qjet.SetBranchAddress("image", qim)
#a=np.array(qim).reshape((3,2*arnum+1,2*arnum+1))
print qjet.GetEntries()
qent=qjet.GetEntries()
gent=gjet.GetEntries()
maxlen=0
for i in range(gjet.GetEntries()/10):
  qjet.GetEntry(i+0)
  gjet.GetEntry(i+0)
  if(len(qjet.dau_pt)>maxlen):maxlen=len(qjet.dau_pt)
  dausort=sorted(range(len(qjet.dau_pt)),key=lambda k: qjet.dau_pt[k],reverse=True)
  #qpt.Fill(len(qjet.dau_pt))
  for j in range(len(qjet.dau_pt)):
    if(j<lg):
      #qpt[j]=qpt[j]+qjet.dau_pt[dausort[j]]/(1.*qjet.GetEntries())
      qpt.Fill(j,qjet.dau_pt[dausort[j]]/(1.*qent))
  if(len(gjet.dau_pt)>maxlen):maxlen=len(gjet.dau_pt)
  dausort=sorted(range(len(gjet.dau_pt)),key=lambda k: gjet.dau_pt[k],reverse=True)
  #gpt.Fill(len(gjet.dau_pt))
  for j in range(len(gjet.dau_pt)):
    if(j<lg):
      #qpt[j]=qpt[j]+qjet.dau_pt[dausort[j]]/(1.*qjet.GetEntries())
      gpt.Fill(j,gjet.dau_pt[dausort[j]]/(1.*gent))
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
#qpt.Fit("expo")
#gpt.Fit("expo")
print(maxlen)
