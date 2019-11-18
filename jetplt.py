from collections import OrderedDict
import numpy as np
import matplotlib as mpl
mpl.rcParams['hatch.linewidth']=2.0
import matplotlib.pyplot as plt
fs=25
plt.rc('xtick',labelsize=int(fs*.8))
plt.rc('ytick',labelsize=int(fs*.8))
import ROOT as rt
from ROOT import TH1F,TColor
import os
# FIXME SetYTitle

class BinaryClassifierResponse(object):
    def __init__(self,
                 name,
                 title,
                 directory,
                 pt):
        f=np.load("jj25{}.npz".format(pt)) 
        self.bdt=f["bdtset"]
        #cmult,nmult,ptd,axis1,axis2,
        #cmult,nmult,ptd,axis1,axis2,
        #dr,pt/pt,mult/mult
        self.eve=f["eveset"]
        self.pairlist=f["pairlist"]

        self._name = name
        self._title = title
        self._path = os.path.join(directory, name + ".{ext}")


    def append(self, pt,var):
        res=30
        maxv=self.bdt[:,var].max()
        if(maxv<self.bdt[:,var].max()):
          maxv=self.bdt[:,var].max()
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
          #q1hist=rt.TH1F("q1hist","pp#rightarrowqq", res, -1,1)
          #q2hist=rt.TH1F("q2hist","Z+jet", res, -1,1)
          #g1hist=rt.TH1F("g1hist","dijet", res, -1,1)
          #g2hist=rt.TH1F("g2hist","Z+jet", res, -1,1)
          q1hist=rt.TH1F("q1hist","pp#rightarrowqq", res, -2.4,2.4)
          q2hist=rt.TH1F("q2hist","Z+jet", res, -2.4,2.4)
          g1hist=rt.TH1F("g1hist","dijet", res, -2.4,2.4)
          g2hist=rt.TH1F("g2hist","Z+jet", res, -2.4,2.4)
        elif(var=="pt"):
          q1hist=rt.TH1F("q1hist","dijet", res, 0, maxpt)
          q2hist=rt.TH1F("q2hist","Z+jet", res, 0, maxpt)
          g1hist=rt.TH1F("g1hist","dijet", res, 0, maxpt)
          g2hist=rt.TH1F("g2hist","Z+jet", res, 0, maxpt)
        else:
          maxv=int(0.8*maxv)
          vb=int(maxv)
          #if(pt==500 or pt==1000):vb=int(vb/2)
          #vb=res
          print("max",maxv)
          q1hist=rt.TH1F("q1hist","dijet", vb,0, maxv)
          q2hist=rt.TH1F("q2hist","Z+jet", vb,0, maxv)
          g1hist=rt.TH1F("g1hist","dijet", vb,0, maxv)
          g2hist=rt.TH1F("g2hist","Z+jet", vb,0, maxv)
          res=int(vb)
          for k in range(len(self.bdt)):
            pair=np.argmax(self.eve[k])
            pair=self.pairlist[np.argmax(self.eve[k])]
            """if(pair==0):
              q1hist.Fill(self.bdt[k][var])
            if(pair==1):
              q2hist.Fill(self.bdt[k][var])
            if(pair==2):
              g1hist.Fill(self.bdt[k][var])
            if(pair==3):
              g2hist.Fill(self.bdt[k][var])"""
            if(pair[0]=="q"):
              q1hist.Fill(self.bdt[k][0])
            if(pair[1]=="q"):
              q2hist.Fill(self.bdt[k][5])
            if(pair[0]=="g"):
              g1hist.Fill(self.bdt[k][0])
            if(pair[1]=="g"):
              g2hist.Fill(self.bdt[k][5])
        q1hist.Scale(1.0 / q1hist.Integral())
        #q2hist.Scale(1.0 / q2hist.Integral())
        g1hist.Scale(1.0 / g1hist.Integral())
        #g2hist.Scale(1.0 / g2hist.Integral())
        q1hist.Draw("hist")
        q1=[]
        q2=[]
        g1=[]
        g2=[]

        for i in range(res):
          q1.append(q1hist.GetBinContent(i+1))
          q2.append(q2hist.GetBinContent(i+1))
          g1.append(g1hist.GetBinContent(i+1))
          g2.append(g2hist.GetBinContent(i+1))
        print(g2hist.GetEntries())
        fs=25
        plt.figure(figsize=(12,8))   
        plt.title("jet $p_T$ range {}~{} GeV".format(pt,int(pt*1.1)),fontdict={"weight":"bold","size":fs*1.})
        plt.ylabel("Fraction of Events",fontsize=fs*1.3)
        if(var=='eta'):
          xax=np.append(np.arange(-2.4,2.4,((-q1hist.GetBinLowEdge(0)+q1hist.GetBinLowEdge(res))/res))[:res],2.4)
          plt.xlabel("jet $\eta$",fontsize=fs*1.3)
          plt.fill_between(xax,np.append(q1,0),alpha=0.6,linewidth=2,facecolor='azure',edgecolor='C0',label=r"pp$\rightarrow$qq",step='post')
          plt.plot(xax,np.append(q1,0),color='C0',alpha=0.5,drawstyle='steps-post',linewidth=2)
          plt.plot(xax,np.append(q2,0),label=r"pp$\rightarrow$q2",color='blue',drawstyle='steps-post',linewidth=3)
          plt.fill_between(xax,np.append(g1,0),alpha=0.3,linewidth=2,linestyle='--',facecolor="#ffcfdc",edgecolor='C1',label=r"pp$\rightarrow$gg",step='post')
          #plt.plot(xax,np.append(g1,0),label=r"pp$\rightarrow$gg",color='C1',drawstyle='steps-post',linewidth=3,linestyle='--',alpha=0.6)
          plt.plot(xax,np.append(g2,0),label=r"pp$\rightarrow$g2",color='red',drawstyle='steps-post',linewidth=3,linestyle='--')
        else:
          #if("chad" in var):plt.xlabel("Charged Particle Multiplicity",fontsize=fs*1.)
          #elif("ptd" in var):plt.xlabel("jet $p_TD$",fontsize=fs*1.3)
          #elif("axis" in var):plt.xlabel("jet {}".format(var),fontsize=fs*1.3)
          #else:plt.xlabel("jet $p_T$(GeV)",fontsize=fs*1.3)
          q1=np.concatenate([[0],q1])
          q2=np.concatenate([[0],q2])
          g1=np.concatenate([[0],g1])
          g2=np.concatenate([[0],g2])
          plt.xlabel("Charged Particle Multiplicity",fontsize=fs*1.)
          plt.fill_between(np.linspace(0,q1hist.GetBinLowEdge(res)+1,res+1),q1,alpha=0.6,linewidth=2,facecolor='azure',edgecolor='C0',label=r"Quark jet",step='pre')
          plt.plot(np.linspace(0,q1hist.GetBinLowEdge(res)+1,res+1),q1,color='C0',alpha=0.6,drawstyle='steps',linewidth=2)
          #plt.plot(np.linspace(0,q2hist.GetBinLowEdge(res)+1,res+1),q2,label=r"pp$\rightarrow$qg",drawstyle='steps',linewidth=3,linestyle="--")
          plt.fill_between(np.linspace(0,g1hist.GetBinLowEdge(res)+1,res+1),g1,facecolor="#ffcfdc",edgecolor='red',alpha=0.4,linewidth=2,label=r"Gluon jet",step='pre',linestyle='-')
          #plt.plot(np.linspace(0,g2hist.GetBinLowEdge(res)+1,res+1),g2,color='red',label=r"pp$\rightarrow$gg",drawstyle='steps',linewidth=3,linestyle='--')
        plt.grid(alpha=0.6)
        pos=0
        a1,a2,b1,b2=plt.axis()
        plt.axis((a1,a2,0,b2))
        if(var=='eta'):
          pos=8 
          plt.axis((-2.4,2.4,0,b2))
        plt.legend(fontsize=fs*1,loc=pos)
        
        plt.savefig(filename+".png",bbox_inches='tight',pad_inches=0.5,dpi=300)
        print(self.pairlist)
        plt.show()
        #plt.savefig(filename+".pdf",bbox_inches='tight',pad_inches=0.5,dpi=300)

#pts=[100]
#pts=[100,200,500,1000]
#pts=[100,200,500,1000]
pts=[200]
#varl=["eta","pt","ptd","major_axis","chad_mult"]
#
#varl=["chad_mult"]
varl=[0]
for pt in pts:
  for var in varl:
    filename="plots/dual{}{}".format(var,pt)
    a= BinaryClassifierResponse(filename,"{}~{}GeV".format(pt,int(pt*1.1)),"./",pt)
    a.append(pt,var)
