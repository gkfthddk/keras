import numpy as np
import matplotlib.pyplot as plt
import copy
import ROOT as rt
pts=[100,200,500,1000]
events=['jj','qq','gg','qg','zj','zq','zg']
cut="pt>{} && pt < {} && eta<1 && eta > -1"
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
    cross[ev][i]=cross[ev][i]/entries[ev][i]*cutentries[ev][i]
    f.Close()
plt.figure(figsize=(12, 8))
plt.ylabel("Effective Cross Section(pb)",fontsize=fs)
plt.xlabel("$p_T$ Range(GeV)",fontsize=fs*1.3)
plt.plot([105,210,525,1050],cross['jj'],
          ':',linewidth=3,label=r"pp$\rightarrow$jj",marker='o',
          alpha=0.7,color='C2',markersize=fs)
plt.plot([105,210,525,1050],cross['qq'],
          ':',linewidth=3,label=r"pp$\rightarrow$qq",marker='v',
          alpha=0.7,color='C0',markersize=fs)
plt.plot([105,210,525,1050],cross['gg'],
          ':',linewidth=3,label=r"pp$\rightarrow$gg",marker='^',
          alpha=0.7,color='C1',markersize=fs)
plt.plot([105,210,525,1050],cross['qg'],
          ':',linewidth=3,label=r"pp$\rightarrow$qg",marker='d',
          alpha=0.7, color='C3',markersize=fs*0.75)
plt.plot([105,210,525,1050],cross['zj'],
          ':',linewidth=3,label=r"pp$\rightarrow$zj",marker='o',
          fillstyle='none',color='C0',markersize=fs*0.75)
plt.plot([105,210,525,1050],cross['zq'],
          ':',linewidth=3,label=r"pp$\rightarrow$zq",marker='v',
          fillstyle='none',color='C1',markersize=fs)
plt.plot([105,210,525,1050],cross['zg'],
          ':',linewidth=3,label=r"pp$\rightarrow$zg",marker='^',
          fillstyle='none',color='C2',markersize=fs)
plt.xticks([105,210,525,1050],["100\n~110","200\n~220","500\n~550","1000\n~1100"],size=fs*0.8)
plt.yticks(size=fs)
#plt.yticks([100*0.05,100*0.06,100*0.07,100*0.08,100*0.09,100*0.10],size=fs)
plt.yscale('log')
plt.grid(alpha=0.6)
plt.legend(fontsize=fs*0.8,ncol=2)
plt.savefig("effective.pdf",bbox_inches='tight',pad_inches=0.5,dpi=300)
plt.savefig("effective.png",bbox_inches='tight',pad_inches=0.5,dpi=300)

plt.figure(figsize=(12, 8))
plt.ylabel("Quark Fraction(%)",fontsize=fs)
plt.xlabel("$p_T$ Range(GeV)",fontsize=fs*1.3)
dfr=(np.array(cross['qq'])+0.5*np.array(cross['qg']))/np.array(cross['jj'])
zfr=(np.array(cross['zq']))/np.array(cross['zj'])
plt.plot([105,210,525,1050],zfr,
          '--',linewidth=3,label=r"Z+jet",marker='v',
          alpha=0.7,color='C1',markersize=fs)
plt.plot([105,210,525,1050],dfr,
          '--',linewidth=3,label=r"dijet",marker='^',
          alpha=0.7,color='C0',markersize=fs)
plt.xticks([105,210,525,1050],["100\n~110","200\n~220","500\n~550","1000\n~1100"],size=fs*0.8)
a1,a2,b1,b2=plt.axis()
plt.axis((a1,a2,0,1))
plt.yticks(size=fs)
print(zfr-dfr)
#plt.yticks([100*0.05,100*0.06,100*0.07,100*0.08,100*0.09,100*0.10],size=fs)
plt.grid(alpha=0.6)
plt.legend(fontsize=fs,loc=4)
plt.savefig("ptetafraction.pdf",bbox_inches='tight',pad_inches=0.5,dpi=300)
plt.savefig("ptetafraction.png",bbox_inches='tight',pad_inches=0.5,dpi=300)
plt.show()
