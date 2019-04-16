from collections import OrderedDict
import numpy as np
import matplotlib as mpl
mpl.rcParams['hatch.linewidth']=2.0
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
        print(pt)
        if(pt==100):maxpt=175
        if(pt==200):maxpt=357
        if(pt==500):maxpt=874
        if(pt==1000):maxpt=1750
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
        
        dqhist.Scale(1.0 / dqhist.Integral())
        zqhist.Scale(1.0 / zqhist.Integral())
        dghist.Scale(1.0 / dghist.Integral())
        zghist.Scale(1.0 / zghist.Integral())
        dqhist.Draw("hist")
        dq=[]
        zq=[]
        dg=[]
        zg=[]

        for i in range(res):
          dq.append(dqhist.GetBinContent(i+1))
          zq.append(zqhist.GetBinContent(i+1))
          dg.append(dghist.GetBinContent(i+1))
          zg.append(zghist.GetBinContent(i+1))
        print(zghist.GetEntries())
        fs=25
        plt.figure(figsize=(12,8))   
        plt.title("jet $p_T$ range {}~{}GeV".format(pt,int(pt*1.1)),fontdict={"weight":"bold","size":fs*1.})
        plt.ylabel("Fraction of Events",fontsize=fs*1.3)
        if(var=='pt'):
          plt.xlabel("jet $p_T$(GeV)",fontsize=fs*1.3)
          plt.fill_between(np.arange(0,dqhist.GetBinLowEdge(res)+1,(dqhist.GetBinLowEdge(res)/res))[:res],dq,alpha=0.4,linewidth=2,facecolor='azure',edgecolor='C0',label=r"pp$\rightarrow$qq",step='pre')
          plt.plot(np.arange(0,dqhist.GetBinLowEdge(res)+1,(dqhist.GetBinLowEdge(res)/res))[:res],dq,color='C0',alpha=0.5,drawstyle='steps',linewidth=2)
          plt.plot(np.arange(0,zqhist.GetBinLowEdge(res)+1,(zqhist.GetBinLowEdge(res)/res))[:res],zq,label=r"pp$\rightarrow$zq",drawstyle='steps',linewidth=3)
          plt.fill_between(np.arange(0,dghist.GetBinLowEdge(res)+1,(dghist.GetBinLowEdge(res)/res))[:res],dg,facecolor="#fff9d0",edgecolor='C1',alpha=0.7,linewidth=2,label=r"pp$\rightarrow$gg",step='pre',linestyle='--')
          plt.plot(np.arange(0,zghist.GetBinLowEdge(res)+1,(zghist.GetBinLowEdge(res)/res))[:res],zg,color='red',label=r"pp$\rightarrow$zg",drawstyle='steps',linewidth=3,linestyle='--')
        if(var=='eta'):
          xax=np.append(np.arange(-2.4,2.4,((-dqhist.GetBinLowEdge(0)+dqhist.GetBinLowEdge(res))/res))[:res],2.4)
          plt.xlabel("jet $\eta$",fontsize=fs*1.3)
          plt.fill_between(xax,np.append(dq,0),alpha=0.4,linewidth=2,facecolor='azure',edgecolor='C0',label=r"pp$\rightarrow$qq",step='post')
          plt.plot(xax,np.append(dq,0),color='C0',alpha=0.5,drawstyle='steps-post',linewidth=2)
          plt.plot(xax,np.append(zq,0),label=r"pp$\rightarrow$zq",color='blue',drawstyle='steps-post',linewidth=3)
          plt.fill_between(xax,np.append(dg,0),alpha=0.7,linewidth=2,linestyle='--',facecolor="#fff9d0",edgecolor='C1',label=r"pp$\rightarrow$gg",step='post')
          #plt.plot(xax,np.append(dg,0),label=r"pp$\rightarrow$gg",color='C1',drawstyle='steps-post',linewidth=3,linestyle='--',alpha=0.6)
          plt.plot(xax,np.append(zg,0),label=r"pp$\rightarrow$zg",color='red',drawstyle='steps-post',linewidth=3,linestyle='--')
        plt.grid(alpha=0.6)
        pos=0
        a1,a2,b1,b2=plt.axis()
        plt.axis((a1,a2,0,b2))
        if(var=='eta'):
          pos=8 
          plt.axis((-2.4,2.4,0,b2))
        plt.legend(fontsize=fs*1,loc=pos)
        
        plt.savefig(filename+".png",bbox_inches='tight',pad_inches=0.5,dpi=300)
        plt.savefig(filename+".pdf",bbox_inches='tight',pad_inches=0.5,dpi=300)

#pts=[100]
pts=[100,200,500,1000]
varl=["eta","pt"]
for pt in pts:
  for var in varl:
    filename="plots/jet{}{}".format(var,pt)
    a= BinaryClassifierResponse(filename,"{}~{}GeV".format(pt,int(pt*1.1)),"./")
    a.append(pt,var)
