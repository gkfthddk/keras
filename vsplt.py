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
        
        zqfile=rt.TFile("Data/zq_pt_{}_{}.root".format(pt,int(pt*1.1)))
        dqfile=rt.TFile("Data/qq_pt_{}_{}.root".format(pt,int(pt*1.1)))
        zgfile=rt.TFile("Data/zg_pt_{}_{}.root".format(pt,int(pt*1.1)))
        dgfile=rt.TFile("Data/gg_pt_{}_{}.root".format(pt,int(pt*1.1)))
        dqjet=dqfile.Get("jetAnalyser")
        zqjet=zqfile.Get("jetAnalyser")
        dgjet=dgfile.Get("jetAnalyser")
        zgjet=zgfile.Get("jetAnalyser")
        maxv=dqjet.GetMaximum(var)
        if(maxv<zqjet.GetMaximum(var)):
          maxv=zqjet.GetMaximum(var)
        if(maxv<zgjet.GetMaximum(var)):
          maxv=zgjet.GetMaximum(var)
        if(maxv<dgjet.GetMaximum(var)):
          maxv=dgjet.GetMaximum(var)
        """if("chad" in var):
          maxv=dqjet.GetMaximum("chad_mult")+dqjet.GetMaximum("electron_mult")+dqjet.GetMaximum("muon_mult")
          if(maxv<zqjet.GetMaximum(var)):
            maxv=zqjet.GetMaximum(var)
          if(maxv<zgjet.GetMaximum(var)):
            maxv=zgjet.GetMaximum(var)
          if(maxv<dgjet.GetMaximum(var)):
            maxv=dgjet.GetMaximum(var)"""
        print(pt)
        if(pt==100):maxpt=175
        if(pt==200):maxpt=357
        if(pt==500):maxpt=874
        if(pt==1000):maxpt=1750
        if(pt==100): 
          ptmin=0.815*pt 
          ptmax=1.159*pt 
        if(pt==200): 
          ptmin=0.819*pt 
          ptmax=1.123*pt 
        if(pt==500): 
          ptmin=0.821*pt 
          ptmax=1.093*pt 
        if(pt==1000): 
          ptmin=0.8235*pt 
          ptmax=1.076*pt
        if(var=="eta"):
          #dqhist=rt.TH1F("dqhist","pp#rightarrowqq", res, -1,1)
          #zqhist=rt.TH1F("zqhist","Z+jet", res, -1,1)
          #dghist=rt.TH1F("dghist","dijet", res, -1,1)
          #zghist=rt.TH1F("zghist","Z+jet", res, -1,1)
          dqhist=rt.TH1F("dqhist","pp#rightarrowqq", res, -2.4,2.4)
          zqhist=rt.TH1F("zqhist","Z+jet", res, -2.4,2.4)
          dghist=rt.TH1F("dghist","dijet", res, -2.4,2.4)
          zghist=rt.TH1F("zghist","Z+jet", res, -2.4,2.4)
        elif(var=="pt"):
          dqhist=rt.TH1F("dqhist","dijet", res, 0, maxpt)
          zqhist=rt.TH1F("zqhist","Z+jet", res, 0, maxpt)
          dghist=rt.TH1F("dghist","dijet", res, 0, maxpt)
          zghist=rt.TH1F("zghist","Z+jet", res, 0, maxpt)
        elif("chad" in var):
          maxv=int(0.8*maxv)
          vb=int(maxv)
          if(pt==500 or pt==1000):vb=int(vb/2)
          print("max",maxv)
          dqhist=rt.TH1F("dqhist","dijet", vb,0, maxv)
          zqhist=rt.TH1F("zqhist","Z+jet", vb,0, maxv)
          dghist=rt.TH1F("dghist","dijet", vb,0, maxv)
          zghist=rt.TH1F("zghist","Z+jet", vb,0, maxv)
          res=int(vb)
        else:
          dqhist=rt.TH1F("dqhist","dijet", res, 0, maxv)
          zqhist=rt.TH1F("zqhist","Z+jet", res, 0, maxv)
          dghist=rt.TH1F("dghist","dijet", res, 0, maxv)
          zghist=rt.TH1F("zghist","Z+jet", res, 0, maxv)
        if("chad" in var):
          for k in range(dqjet.GetEntries()):
            dqjet.GetEntry(k)
            if(dqjet.pt<ptmin or dqjet.pt>ptmax):continue
            dqhist.Fill(dqjet.chad_mult+dqjet.electron_mult+dqjet.muon_mult)
          for k in range(zqjet.GetEntries()):
            zqjet.GetEntry(k)
            if(zqjet.pt<ptmin or zqjet.pt>ptmax):continue
            zqhist.Fill(zqjet.chad_mult+zqjet.electron_mult+zqjet.muon_mult)
          for k in range(dgjet.GetEntries()):
            dgjet.GetEntry(k)
            if(dgjet.pt<ptmin or dgjet.pt>ptmax):continue
            dghist.Fill(dgjet.chad_mult+dgjet.electron_mult+dgjet.muon_mult)
          for k in range(zgjet.GetEntries()):
            zgjet.GetEntry(k)
            if(zgjet.pt<ptmin or zgjet.pt>ptmax):continue
            zghist.Fill(zgjet.chad_mult+zgjet.electron_mult+zgjet.muon_mult)
        else:
          dqjet.Draw(var+">>dqhist","pt<{} && pt >{}".format(ptmax,ptmin))
          zqjet.Draw(var+">>zqhist","pt<{} && pt >{}".format(ptmax,ptmin))
          dgjet.Draw(var+">>dghist","pt<{} && pt >{}".format(ptmax,ptmin))
          zgjet.Draw(var+">>zghist","pt<{} && pt >{}".format(ptmax,ptmin))
         
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
        if(var=='eta'):
          xax=np.append(np.arange(-2.4,2.4,((-dqhist.GetBinLowEdge(0)+dqhist.GetBinLowEdge(res))/res))[:res],2.4)
          plt.xlabel("jet $\eta$",fontsize=fs*1.3)
          plt.fill_between(xax,np.append(dq,0),alpha=0.6,linewidth=2,facecolor='azure',edgecolor='C0',label=r"pp$\rightarrow$qq",step='post')
          plt.plot(xax,np.append(dq,0),color='C0',alpha=0.5,drawstyle='steps-post',linewidth=2)
          plt.plot(xax,np.append(zq,0),label=r"pp$\rightarrow$zq",color='blue',drawstyle='steps-post',linewidth=3)
          plt.fill_between(xax,np.append(dg,0),alpha=0.3,linewidth=2,linestyle='--',facecolor="#ffcfdc",edgecolor='C1',label=r"pp$\rightarrow$gg",step='post')
          #plt.plot(xax,np.append(dg,0),label=r"pp$\rightarrow$gg",color='C1',drawstyle='steps-post',linewidth=3,linestyle='--',alpha=0.6)
          plt.plot(xax,np.append(zg,0),label=r"pp$\rightarrow$zg",color='red',drawstyle='steps-post',linewidth=3,linestyle='--')
        else:
          if("chad" in var):plt.xlabel("Charged particle multiplicity",fontsize=fs*1.)
          elif("ptd" in var):plt.xlabel("jet $p_TD$",fontsize=fs*1.3)
          elif("axis" in var):plt.xlabel("jet {}".format(var),fontsize=fs*1.3)
          else:plt.xlabel("jet $p_T$(GeV)",fontsize=fs*1.3)
          plt.fill_between(np.arange(0,dqhist.GetBinLowEdge(res)+1,(dqhist.GetBinLowEdge(res)/res))[:res],dq,alpha=0.6,linewidth=2,facecolor='azure',edgecolor='C0',label=r"pp$\rightarrow$qq",step='pre')
          plt.plot(np.arange(0,dqhist.GetBinLowEdge(res)+1,(dqhist.GetBinLowEdge(res)/res))[:res],dq,color='C0',alpha=0.5,drawstyle='steps',linewidth=2)
          plt.plot(np.arange(0,zqhist.GetBinLowEdge(res)+1,(zqhist.GetBinLowEdge(res)/res))[:res],zq,label=r"pp$\rightarrow$zq",drawstyle='steps',linewidth=3)
          plt.fill_between(np.arange(0,dghist.GetBinLowEdge(res)+1,(dghist.GetBinLowEdge(res)/res))[:res],dg,facecolor="#ffcfdc",edgecolor='C1',alpha=0.3,linewidth=2,label=r"pp$\rightarrow$gg",step='pre',linestyle='--')
          plt.plot(np.arange(0,zghist.GetBinLowEdge(res)+1,(zghist.GetBinLowEdge(res)/res))[:res],zg,color='red',label=r"pp$\rightarrow$zg",drawstyle='steps',linewidth=3,linestyle='--')
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
#pts=[100,200,500,1000]
pts=[100,200,500,1000]
#pts=[100,1000]
varl=["eta","pt","ptd","major_axis","chad_mult"]
#varl=["chad_mult"]
for pt in pts:
  for var in varl:
    filename="plots/jet{}{}".format(var,pt)
    a= BinaryClassifierResponse(filename,"{}~{}GeV".format(pt,int(pt*1.1)),"./")
    a.append(pt,var)
