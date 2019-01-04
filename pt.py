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
qf=rt.TFile("sdata/dijet_{0}_{1}/dijet_{0}_{1}_training.root".format(pt,int(pt*1.1)),'read')
qjet=qf.Get("jetAnalyser")
arnum=16
qpt=rt.TH1F("qpt","quark",20,0,20)
gpt=rt.TH1F("gpt","gluon",20,0,20)
#qim = array('B', [0]*(3*(arnum*2+1)*(arnum*2+1)))
#qjet.SetBranchAddress("image", qim)
#a=np.array(qim).reshape((3,2*arnum+1,2*arnum+1))
print qjet.GetEntries()
qent=qjet.GetEntries("label==1")
gent=qjet.GetEntries("label==0")
for i in range(qjet.GetEntries()/100):
  qjet.GetEntry(i+0)
  if(qjet.label==1):
    dausort=sorted(range(len(qjet.dau_pt)),key=lambda k: qjet.dau_pt[k],reverse=True)
    for j in range(len(qjet.dau_pt)):
      if(j<20):
        #qpt[j]=qpt[j]+qjet.dau_pt[dausort[j]]/(1.*qjet.GetEntries())
        qpt.Fill(j,qjet.dau_pt[dausort[j]]/(1.*qent))
  else:
    dausort=sorted(range(len(qjet.dau_pt)),key=lambda k: qjet.dau_pt[k],reverse=True)
    for j in range(len(qjet.dau_pt)):
      if(j<20):
        #qpt[j]=qpt[j]+qjet.dau_pt[dausort[j]]/(1.*qjet.GetEntries())
        gpt.Fill(j,qjet.dau_pt[dausort[j]]/(1.*gent))
canv=rt.TCanvas("canv","Paticles' mean p_{T} by ordr in each jets",1000,700)
canv.cd(1)
gStyle.SetOptStat(0)
qpt.SetLineColor(4)
qpt.SetLineWidth(3)
qpt.GetXaxis().SetTitle("Order")
qpt.GetYaxis().SetTitle("Mean p_{T} GeV")
qpt.Draw("HIST")
gpt.SetLineColor(2)
gpt.SetLineWidth(3)
gpt.GetXaxis().SetTitle("Order")
gpt.GetYaxis().SetTitle("Mean p_{T} GeV")
gpt.Draw("HIST Same")
gPad.BuildLegend()
qpt.SetTitle("Mean p_{T} by odering at jet p_{T} "+"{0}~{1}".format(pt,int(pt*1.1)))
#plt.plot(range(20),gpt,range(20),qpt)
#plt.show()
qpt.Fit("expo")
gpt.Fit("expo")
