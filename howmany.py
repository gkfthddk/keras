import os
import sys
import subprocess
import ROOT as rt
import numpy as np
ls=subprocess.check_output("ls Data/*.root",shell=True)
ls=ls.split("\n")[:-1]

for l in ls:
  f=rt.TFile(l,"read")
  a=np.load(l[:-4]+"npz")
  print(l,f.Get("jetAnalyser").GetEntries(),a["chad_mult"].shape)
  #print(l,f.Get("jetAnalyser").GetEntries("eta<1&&eta>-1"))
  f.Close()
  a.close()
