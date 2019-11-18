import os
import sys
import subprocess
import ROOT as rt
import numpy as np
import scipy.stats
ptmin=0
ptmax=0
frac={"qq":[],"gg":[],"qg":[],"zq":[],"zg":[],"jj":[],"zj":[]}
frac2={"qq":[],"gg":[],"qg":[],"zq":[],"zg":[],"jj":[],"zj":[]}
frac3={"qq":[],"gg":[],"qg":[],"zq":[],"zg":[],"jj":[],"zj":[]}
for pt in [100,200,500,1000]:
  ls=subprocess.check_output("ls Data/*_{}_*.root".format(pt),shell=True)
  ls=ls.split("\n")[:-1]
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
  for l in ls:
    #if("qg" in ls):continue
    f=rt.TFile(l,"read")
    a=np.load(l[:-4]+"npz")
    #print(l,f.Get("jetAnalyser").GetEntries(),a["chad_mult"].shape)
    matchedg=f.Get("jetAnalyser").GetEntries("pt<{}&&pt>{}&&parton_id==21".format(ptmax,ptmin))
    noptmatchedg=f.Get("jetAnalyser").GetEntries("parton_id==21".format(ptmax,ptmin))
    matchedq=f.Get("jetAnalyser").GetEntries("pt<{}&&pt>{}&&parton_id!=21&&parton_id!=0".format(ptmax,ptmin))
    noptmatchedq=f.Get("jetAnalyser").GetEntries("parton_id!=21&&parton_id!=0".format(ptmax,ptmin))
    unmatched=f.Get("jetAnalyser").GetEntries("pt<{}&&pt>{}&&eta<1&&eta>-1&&parton_id==0".format(ptmax,ptmin))
    sem=scipy.stats.sem(np.array(([1]*matchedq)+([0]*matchedg)))
    sem2=scipy.stats.sem(np.array(([1]*(matchedq+matchedg))+([0]*unmatched)))
    ptcut=f.Get("jetAnalyser").GetEntries("pt<{}&&pt>{}".format(ptmax,ptmin))
    print(l,ptcut,f.Get("jetAnalyser").GetEntries("pt<{}&&pt>{}&&eta<1&&eta>-1".format(ptmax,ptmin)),1.*matchedq/(matchedg+matchedq),1.*noptmatchedq/(noptmatchedg+noptmatchedq),1-1.*unmatched/ptcut,sem)
    frac[l[5:7]].append(round(1-1.*unmatched/ptcut,3))
    frac2[l[5:7]].append(round(sem,3))
    frac3[l[5:7]].append(round(sem2,3))
    f.Close()
    a.close()
for k in frac:
  a=" {} |"*5
  a="|"+a
  print(a.format(k,*frac[k]))
for k in frac2:
  a=" {} |"*5
  a="|"+a
  print(a.format(k,*frac2[k]))
for k in frac3:
  a=" {} |"*5
  a="|"+a
  print(a.format(k,*frac3[k]))
