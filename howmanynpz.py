import os
import sys
import subprocess
import ROOT as rt
import numpy as np
ptmin=0
ptmax=0
for pt in [100,200,500,1000]:
  ls=subprocess.check_output("ls *mixed*{}p*.npz".format(pt),shell=True)
  ls=ls.split("\n")[:-1]
  for l in ls:
    if("unscale" in l):continue
    a=np.load(l[:])
    #print(l,f.Get("jetAnalyser").GetEntries(),a["chad_mult"].shape)
    print(l,len(a["ptset"]))
    a.close()
