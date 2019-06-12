from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import ROOT as rt
from ROOT import TH1F,TColor
import os

# FIXME SetYTitle

class BinaryClassifierResponse(object):
    def __init__(self,
                 name,
                 title,
                 directory):

        self._name = name
        self._title = title
        self._path = os.path.join(directory, name + ".{ext}")


    def append(self, pt,var):
        res=30
        
        dqfile=rt.TFile("Data/zq_pt_{}_{}.root".format(pt,int(pt*1.1)))
        zqfile=rt.TFile("Data/qq_pt_{}_{}.root".format(pt,int(pt*1.1)))
        dgfile=rt.TFile("Data/zg_pt_{}_{}.root".format(pt,int(pt*1.1)))
        zgfile=rt.TFile("Data/gg_pt_{}_{}.root".format(pt,int(pt*1.1)))
        dqjet=dqfile.Get("jetAnalyser")
        zqjet=zqfile.Get("jetAnalyser")
        dgjet=dgfile.Get("jetAnalyser")
        zgjet=zgfile.Get("jetAnalyser")
        maxpt=dqjet.GetMaximum(var)
        if(maxpt<zqjet.GetMaximum(var)):
          maxpt=zqjet.GetMaximum(var)
        if(maxpt<zgjet.GetMaximum(var)):
          maxpt=zgjet.GetMaximum(var)
        if(maxpt<dgjet.GetMaximum(var)):
          maxpt=dgjet.GetMaximum(var)
        if(pt==100):maxpt=200
        if(pt==200):maxpt=350
        if(pt==500):maxpt=800
        if(pt==1000):maxpt=1600
        h0 = TH1F("untitled", self._title, res, -maxpt, maxpt)
        if(var=="eta"):
          dqhist=rt.TH1F("dqhist","pp#rightarrowqq", res, -2.4,2.4)
          zqhist=rt.TH1F("zqhist","Z+jet", res, -2.4,2.4)
          dghist=rt.TH1F("dghist","dijet", res, -2.4,2.4)
          zghist=rt.TH1F("zghist","Z+jet", res, -2.4,2.4)
        if(var=="pt"):
          dqhist=rt.TH1F("dqhist","dijet", res, 0, maxpt)
          zqhist=rt.TH1F("zqhist","Z+jet", res, 0, maxpt)
          dghist=rt.TH1F("dghist","dijet", res, 0, maxpt)
          zghist=rt.TH1F("zghist","Z+jet", res, 0, maxpt)
        dqjet.Draw(var+">>dqhist")
        zqjet.Draw(var+">>zqhist")
        dgjet.Draw(var+">>dghist")
        zgjet.Draw(var+">>zghist")
        
        canvas = rt.TCanvas("c", "c", 1200, 800)
        canvas.cd()

        dqhist.Scale(1.0 / dqhist.Integral())
        zqhist.Scale(1.0 / zqhist.Integral())
        dghist.Scale(1.0 / dghist.Integral())
        zghist.Scale(1.0 / zghist.Integral())
        dqhist.Draw("HIST") 
        dqhist.SetLineColor(4)
        dqhist.SetFillColorAlpha(2,0.5)

        dq=[]
        zq=[]
        dg=[]
        zg=[]

        for i in range(res):
          dq.append(dqhist.GetBinContent(i+1))
          zq.append(zqhist.GetBinContent(i+1))
          dg.append(dghist.GetBinContent(i+1))
          zg.append(zghist.GetBinContent(i+1))
          
        plt.fill_between(np.arange(dqhist.GetBinLowEdge(0),dqhist.GetBinLowEdge(res),(-dqhist.GetBinLowEdge(0)+dqhist.GetBinLowEdge(res))/res),dq,alpha=0.2,label=r"pp$\rightarrow$qq",step='pre')
        #plt.plot(np.arange(dqhist.GetBinLowEdge(0),dqhist.GetBinLowEdge(res),(-dqhist.GetBinLowEdge(0)+dqhist.GetBinLowEdge(res))/res),dq,drawstyle='steps')
        #plt.fill_between(np.arange(zqhist.GetBinLowEdge(0),zqhist.GetBinLowEdge(res),(-zqhist.GetBinLowEdge(0)+zqhist.GetBinLowEdge(res))/res),zq,alpha=0.3,label=r"pp$\rightarrow$zq",step='pre')
        plt.plot(np.arange(zqhist.GetBinLowEdge(0),zqhist.GetBinLowEdge(res),(-zqhist.GetBinLowEdge(0)+zqhist.GetBinLowEdge(res))/res),zq,label=r"pp$\rightarrow$zq",drawstyle='steps')
        plt.fill_between(np.arange(dghist.GetBinLowEdge(0),dghist.GetBinLowEdge(res),(-dghist.GetBinLowEdge(0)+dghist.GetBinLowEdge(res))/res),dg,alpha=0.2,label=r"pp$\rightarrow$gg",step='pre')
        #plt.plot(np.arange(dghist.GetBinLowEdge(0),dghist.GetBinLowEdge(res),(-dghist.GetBinLowEdge(0)+dghist.GetBinLowEdge(res))/res),dg,drawstyle='steps')
        #plt.fill_between(np.arange(zghist.GetBinLowEdge(0),zghist.GetBinLowEdge(res),(-zghist.GetBinLowEdge(0)+zghist.GetBinLowEdge(res))/res),zg,alpha=0.3,label=r"pp$\rightarrow$zg",step='pre')
        plt.plot(np.arange(zghist.GetBinLowEdge(0),zghist.GetBinLowEdge(res),(-zghist.GetBinLowEdge(0)+zghist.GetBinLowEdge(res))/res),zg,label=r"pp$\rightarrow$zg",drawstyle='steps')
        plt.grid(alpha=0.6)
        plt.legend()
        a1,a2,b1,b2=plt.axis()
        plt.axis((a1,a2,0,b2))
        plt.savefig(filename,bbox_inches='tight',pad_inches=0.5)
        rt.gStyle.SetOptStat(False)

        canvas.SaveAs(self._path.format(ext="png"))
        canvas.SaveAs(self._path.format(ext="pdf"))

pts=[1000,200,500,1000]
for pt in pts:
    filename="plots/gluonpt{}".format(pt)
    a= BinaryClassifierResponse(filename,"{}~{}GeV".format(pt,int(pt*1.1)),"./")
    a.append(pt,'pt')
