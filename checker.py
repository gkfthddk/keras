import os
import subprocess
import numpy as np
import datetime
import random
import warnings
import ROOT as rt
import math
import sys
from array import array

f=rt.TFile(sys.argv[1],'read')
jet=f.Get("jetAnalyser")
arnum=16
pt=[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
im = array('B', [0]*(3*(arnum*2+1)*(arnum*2+1)))
#jet.SetBranchAddress("image", im)
#a=np.array(im).reshape((3,2*arnum+1,2*arnum+1))
print jet.GetEntries()
for i in range(jet.GetEntries()):
  jet.GetEntry(i+0)
  #dausort=sorted(range(len(jet.dau_pt)),key=lambda k: jet.dau_pt[k],reverse=True)
  #for j in range(len(jet.dau_pt)):
  #  if(j<20):
  #    pt[j]=pt[j]+jet.dau_pt[dausort[j]]
  
  if(len(jet.dau_pt)==0):
    print i," ", len(jet.dau_pt), jet.pt
    break
print pt
