import os
import sys
import subprocess
import ROOT as rt
ls=subprocess.check_output("ls Data/*",shell=True)
ls=ls.split("\n")[:-1]

for l in ls:
  f=rt.TFile(l,"read")
  print(l,f.Get("jetAnalyser").GetEntries("eta<1&&eta>-1"))
  f.Close()
