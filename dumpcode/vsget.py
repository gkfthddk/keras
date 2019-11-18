from collections import OrderedDict
import numpy as np
import matplotlib as mpl
mpl.rcParams['hatch.linewidth']=2.0
mpl.rcParams['figure.dpi']=150
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


    def append(self, pt):
        res=30
        
        dqfile=[eval(i) for i in open("save/pepzjcnn{}model/v1t3q.dat".format(pt)).readlines()]
        zqfile=[eval(i) for i in open("save/pepzjcnn{}model/v1t2q.dat".format(pt)).readlines()]
        dgfile=[eval(i) for i in open("save/pepzjcnn{}model/v1t3g.dat".format(pt)).readlines()]
        zgfile=[eval(i) for i in open("save/pepzjcnn{}model/v1t2g.dat".format(pt)).readlines()]
        cutmax=3#0.61#0.38
        cutmin=-3#0.5#0.33
        pmax=1.38#0.61
        pmin=0.0#0.5
        cutvar=2
        mid=1
        idx=0 
        if(mid==1):
          buf=[[],[],[]]
          while(idx<len(dgfile[0])):
            if(dgfile[cutvar][idx]<cutmax and dgfile[cutvar][idx]>cutmin):
              if(dgfile[0][idx]<pmax and dgfile[0][idx]>pmin):
                for k in range(3):buf[k].append(dgfile[k][idx])
            idx+=1
          idx=0 
          dgfile=buf
          buf=[[],[],[]]
          while(idx<len(zgfile[0])):
            if(zgfile[cutvar][idx]<cutmax and zgfile[cutvar][idx]>cutmin):
              if(zgfile[0][idx]<pmax and zgfile[0][idx]>pmin):
                for k in range(3):buf[k].append(zgfile[k][idx])
            idx+=1
          zgfile=buf
        if(mid==0):
          while(idx<len(dqfile[0])):
            if(dqfile[cutvar][idx]>cutmax or dqfile[cutvar][idx]<cutmin):
              #pass
              if(dqfile[0][idx]<pmax and dqfile[0][idx]>pmin):
                pass
              else:
                del dqfile[0][idx]
                del dqfile[1][idx]
                del dqfile[2][idx]
                idx-=1
            else:
              del dqfile[0][idx]
              del dqfile[1][idx]
              del dqfile[2][idx]
              idx-=1
            idx+=1
          idx=0 
          while(idx<len(zqfile[0])):
            if(zqfile[cutvar][idx]>cutmax or zqfile[cutvar][idx]<cutmin):
              #pass
              if(zqfile[0][idx]<pmax and zqfile[0][idx]>pmin):
                pass
              else:
                del zqfile[0][idx]
                del zqfile[1][idx]
                del zqfile[2][idx]
                idx-=1
            else:
              del zqfile[0][idx]
              del zqfile[1][idx]
              del zqfile[2][idx]
              idx-=1
            idx+=1
        
        dat=[dqfile,zqfile,dgfile,zgfile]
        lab=["dijet-q","z+jet-q","dijet-g","Z+jet-g"]
        for i in range(3):
          for j in range(2,4):
            if(i==0):
              plt.subplot(121)
            if(i==1):
              plt.subplot(222)
            if(i==2):
              plt.subplot(224)
            plt.hist(dat[j][i],bins=50,density=1,histtype="step",alpha=0.5,label=lab[j])
            #plt.autoscale(1)
            if(i==1):
              a,b,c,d=plt.axis()
              plt.axis((a,pt*2,c,d))
            plt.legend()
            plt.grid(alpha=0.3)
        pos=0
        plt.show() 
        #plt.savefig(filename+".png",bbox_inches='tight',pad_inches=0.5,dpi=300)
        #plt.savefig(filename+".pdf",bbox_inches='tight',pad_inches=0.5,dpi=300)

#pts=[100]
pts=[500,500,200,100]
varl=["eta","pt"]
for pt in pts:
    filename="plots/get{}".format(pt)
    a= BinaryClassifierResponse(filename,"{}~{}GeV".format(pt,int(pt*1.1)),"./")
    a.append(pt)
