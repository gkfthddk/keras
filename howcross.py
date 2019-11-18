import numpy as np
import copy
import ROOT as rt
pts=[100,200,500,1000]
events=['jj','qq','gg','qg','zj','zq','zg']
#cut="pt>{} && pt < {} && eta<1 && eta > -1"
cut="pt>{} && pt < {}"
fs=25
#f=open("etacrosssection")
f=open("effectivecrosssection")
cross=eval("".join(f.readlines()))
f.close()
entries=copy.deepcopy(cross)
cutentries=copy.deepcopy(cross)
for i in range(len(pts)):
  for ev in events:
    pt=pts[i]
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
    f=rt.TFile("Data/{}_pt_{}_{}.root".format(ev,pts[i],int(1.1*pts[i])),'read')
    entries[ev][i]=f.Get("jetAnalyser").GetEntries()
    cutentries[ev][i]=f.Get("jetAnalyser").GetEntries(cut.format(ptmin,ptmax))
    cross[ev][i]=format(cross[ev][i]/entries[ev][i]*cutentries[ev][i],'.2e')
    f.Close()
for k in cross:
    a=" {} |"*5
    a="|"+a
    print(a.format(k,*cross[k]))
