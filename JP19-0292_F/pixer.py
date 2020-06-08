import os
import sys
import subprocess
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import random
import ROOT as rt
ptmin=0
ptmax=0
now=datetime.now()
#for sample in ["zq","qq","zg","gg"]:
for sample in ["jj"]:
  for pt in [200,500]:
    #jname='Data/{}_pt_{}_{}.root'.format(sample,pt,int(pt*1.1))
    jname='genroot/pp_jj_{}.root'.format(pt)
    jfile=rt.TFile(jname,'read')
    jet=jfile.Get('jetAnalyser')
    length=jet.GetEntries()
    if(length>40000):
      length=40000
    jet.GetEntry(0)
    chpt=np.array(jet.image_chad_pt_33)/(1.*max(jet.image_chad_pt_33))
    chmt=np.array(jet.image_chad_mult_33)/(1.*max(jet.image_chad_mult_33))
    for i in xrange(1,length):
      jet.GetEntry(i)
      if(max(jet.image_chad_pt_33)==0):continue
      if(jet.pt<pt*1. or jet.pt>pt*1.2):continue
      bchpt=np.array(jet.image_chad_pt_33)/(1.*max(jet.image_chad_pt_33))
      bchmt=np.array(jet.image_chad_mult_33)/(1.*max(jet.image_chad_mult_33))
      chpt=chpt+bchpt
      chmt=chmt+bchmt
    chpt=chpt/(1.*length)
    chmt=chmt/(1.*length)
    #chpt=chpt.reshape((33,33))
    #plt.imshow(chpt)
    jfile.Close()
    f=open("{}chpt{}".format(sample,pt),"write")
    f.write(str(chpt.tolist()))
    f.close()
    f=open("{}chmt{}".format(sample,pt),"write")
    f.write(str(chmt.tolist()))
    f.close()

