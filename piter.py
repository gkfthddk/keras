import numpy as np
import datetime
import random
import ROOT as rt
from ROOT import gPad,gStyle
import math
import sys
from array import array
import matplotlib.pyplot as plt
for sample in ["zq","qq","zg","gg"]:
  for pt in [100,200,500,1000]:
    jname='Data/{}_pt_{}_{}.root'.format(sample,pt,int(pt*1.1))
    jfile=rt.TFile(jname,'read')
    jet=jfile.Get('jetAnalyser')
    length=jet.GetEntries()
    if(length>40000):
      length=40000
    lg=64
    ptlist=[0.]*lg
    drlist=[0.]*lg
    length=jet.GetEntries()
    if(length>40000):
      length=40000
    for i in xrange(length):
      jet.GetEntry(i)
      maxlen=len(jet.dau_pt)
      dausort=sorted(range(maxlen),key=lambda k: jet.dau_pt[k],reverse=True)
      maxpt=max(jet.dau_pt)
      if(maxpt==0):continue
      for j in range(lg):
        if(j<maxlen):
          ptlist[j]=ptlist[j]+jet.dau_pt[dausort[j]]/(1.*length)
          drlist[j]=drlist[j]+np.sqrt(pow(jet.dau_dphi[dausort[j]],2)+pow(jet.dau_deta[dausort[j]],2))/(1.*length)
        else:
          pass
    jfile.Close()
    f=open("{}ptlist{}".format(sample,pt),"write")
    f.write(str(ptlist))
    f.close()
    f=open("{}drlist{}".format(sample,pt),"write")
    f.write(str(drlist))
    f.close()
