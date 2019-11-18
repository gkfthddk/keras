from collections import OrderedDict
import numpy as np
import matplotlib
matplotlib.use("agg")
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


    def append(self, pts,var):
        res=100
        dq1=[]
        zq1=[]
        dg1=[]
        zg1=[]
        dq2=[]
        zq2=[]
        dg2=[]
        zg2=[]
        data=[]
        a=-1
        for pt in pts:
          dq1.append([])
          zq1.append([])
          dg1.append([])
          zg1.append([])
          a+=1
          zqfile=rt.TFile("Data/zq_pt_{}_{}.root".format(pt,int(pt*1.1)))
          dqfile=rt.TFile("Data/qq_pt_{}_{}.root".format(pt,int(pt*1.1)))
          zgfile=rt.TFile("Data/zg_pt_{}_{}.root".format(pt,int(pt*1.1)))
          dgfile=rt.TFile("Data/gg_pt_{}_{}.root".format(pt,int(pt*1.1)))
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
          """if 1:
            dqhist=rt.TH1F("dqhist","dijet", res, 0, maxpt)
            zqhist=rt.TH1F("zqhist","Z+jet", res, 0, maxpt)
            dghist=rt.TH1F("dghist","dijet", res, 0, maxpt)
            zghist=rt.TH1F("zghist","Z+jet", res, 0, maxpt)
          dqjet.Draw(var+">>dqhist")
          zqjet.Draw(var+">>zqhist")
          dgjet.Draw(var+">>dghist")
          zgjet.Draw(var+">>zghist")
          """
          for i in range(20000):
            dqjet.GetEntry(i)
            zqjet.GetEntry(i)
            dgjet.GetEntry(i)
            zgjet.GetEntry(i)
            if(var=="chad_mult"):
              if(abs(dqjet.eta)<1):dq1[a].append(eval("dqjet."+var)+dqjet.electron_mult+dqjet.muon_mult)
              zq1[a].append(eval("zqjet."+var)+zqjet.electron_mult+zqjet.muon_mult)
              dg1[a].append(eval("dgjet."+var)+dgjet.electron_mult+dgjet.muon_mult)
              zg1[a].append(eval("zgjet."+var)+zgjet.electron_mult+zgjet.muon_mult)
            elif(var=="nhad_mult"):
              dq1[a].append(eval("dqjet."+var)+dqjet.photon_mult)
              zq1[a].append(eval("zqjet."+var)+zqjet.photon_mult)
              dg1[a].append(eval("dgjet."+var)+dgjet.photon_mult)
              zg1[a].append(eval("zgjet."+var)+zgjet.photon_mult)
            elif(var=="pt"):
              dq1[a].append(eval("dqjet."+var))
              zq1[a].append(eval("zqjet."+var))
              dg1[a].append(eval("dgjet."+var))
              zg1[a].append(eval("zgjet."+var))
            else:
              if(abs(eval("dqjet."+var))<2.4):
                dq1[a].append(eval("dqjet."+var))
              if(abs(eval("zqjet."+var))<2.4):
                zq1[a].append(eval("zqjet."+var))
              if(abs(eval("dgjet."+var))<2.4):
                dg1[a].append(eval("dgjet."+var))
              if(abs(eval("zgjet."+var))<2.4):
                zg1[a].append(eval("zgjet."+var))

              #dq1[a].append(eval("dqjet."+var))
              #zq1[a].append(eval("zqjet."+var))
              #dg1[a].append(eval("dgjet."+var))
              #zg1[a].append(eval("zgjet."+var))
          
          """dqhist.Scale(1.0 / dqhist.Integral())
          maxbin=dqhist.GetMaximumBin()
          width=0
          for i in range(30):
            if(dqhist.Integral(maxbin-i,maxbin+i)>=0.68):
              dq1.append(dqhist.GetBinContent(maxbin-i))
              dq2.append(dqhist.GetBinContent(maxbin+i))
              break
          zqhist.Scale(1.0 / zqhist.Integral())
          dghist.Scale(1.0 / dghist.Integral())
          zghist.Scale(1.0 / zghist.Integral())
          dqhist.Draw("hist")
          print(zghist.GetEntries())"""
        fs=25
        plt.figure(figsize=(12,8))   
        #plt.title("{}".format(var),fontdict={"weight":"bold","size":fs*1.})
        if("chad" in var):
          plt.ylabel("Charged particle multiplicity",fontsize=fs)
        if("nhad" in var):
          plt.ylabel("Neutral particle multiplicity",fontsize=fs)
        if("ptd" in var):
          plt.ylabel(r"$p_TD$",fontsize=fs)
        if("major" in var):
          plt.ylabel("Major axis",fontsize=fs)
        if("minor" in var):
          plt.ylabel("Minor axis",fontsize=fs)
        if("pt" == var):
          plt.ylabel(r"$p_T$",fontsize=fs)
        if("eta" == var):
          plt.ylabel(r"$\eta$",fontsize=fs)
        if(1):
          plt.xlabel("jet $p_T$ range(GeV)",fontsize=fs*1)
          #plt.fill_between([100,110],[maxbin-width,0],[maxbin+width,0],alpha=0.4,linewidth=2,facecolor='azure',edgecolor='C0',label=r"pp$\rightarrow$qq",step='post')
          pos=np.array([0.75,1.75,2.75,3.75])
          #bp1=plt.violinplot(dq1,positions=pos,widths=0.13,)
          #bp2=plt.violinplot(zq1,positions=pos+0.15,widths=0.13)
          #bp3=plt.violinplot(dg1,positions=pos+0.3,widths=0.13)
          #bp4=plt.violinplot(zg1,positions=pos+0.45,widths=0.13)
          bp1=plt.boxplot(dq1,positions=pos,widths=0.1,patch_artist=1,showfliers=0,bootstrap=1)
          bp2=plt.boxplot(zq1,positions=pos+0.15,widths=0.1,patch_artist=1,showfliers=0,bootstrap=1)
          bp3=plt.boxplot(dg1,positions=pos+0.3,widths=0.1,patch_artist=1,showfliers=0,bootstrap=1)
          bp4=plt.boxplot(zg1,positions=pos+0.45,widths=0.1,patch_artist=1,showfliers=0,bootstrap=1)
          bed='boxes'
          for j in range(4):
            bp2[bed][j].set_linestyle('-')
            bp2[bed][j].set_linewidth(2)
            bp2[bed][j].set_edgecolor('black')
            bp2[bed][j].set_facecolor('white')
            bp4[bed][j].set_linestyle('-')
            bp4[bed][j].set_linewidth(2)
            bp4[bed][j].set_edgecolor('black')
            bp4[bed][j].set_facecolor('#cfe3f9')

            bp1[bed][j].set_linestyle('--')
            bp1[bed][j].set_linewidth(2)
            bp1[bed][j].set_edgecolor('black')
            bp1[bed][j].set_facecolor('white')
            bp3[bed][j].set_linestyle('--')
            bp3[bed][j].set_linewidth(2)
            bp3[bed][j].set_edgecolor('black')
            bp3[bed][j].set_facecolor('#cfe3f9')
          plt.xticks([1,2,3,4],["100~110","200~220","500~550","1000~1100"],size=fs)
          plt.yticks(size=fs)
          #plt.plot(np.arange(0,zghist.GetBinLowEdge(res)+1,(zghist.GetBinLowEdge(res)/res))[:res],zg,color='red',label=r"pp$\rightarrow$zg",drawstyle='steps',linewidth=3,linestyle='--')
        plt.grid(alpha=0.3,axis="y")
        pos=0
        a1,a2,b1,b2=plt.axis()
        plt.axis((0.5,4.5,b1,b2))
        if(var=='eta'):
          pos=8 
          #plt.axis((-2.4,2.4,0,b2))
        plt.legend([bp1[bed][0],bp2[bed][0],bp3[bed][0],bp4[bed][0]],[r'pp$\rightarrow$qq',r'pp$\rightarrow$zq',r'pp$\rightarrow$gg',r'pp$\rightarrow$zg'],fontsize=fs*1,loc=pos,ncol=2)
        plt.savefig(filename+".png",bbox_inches='tight',pad_inches=0.5,dpi=300)
        plt.savefig(filename+".pdf",bbox_inches='tight',pad_inches=0.5,dpi=300)
        #plt.show()

#pts=[100]
pts=[100,200,500,1000]
varl=["eta","nhad_mult","chad_mult","minor_axis","major_axis",'ptd',"pt"]
#varl=["eta"]
for var in varl:
    filename="plots/jet{}".format(var)
    if("chad" in var):
      filename="plots/jetcmult"
    if("nhad" in var):
      filename="plots/jetnmult"
    a= BinaryClassifierResponse(filename,"GeV","./")
    a.append(pts,var)
