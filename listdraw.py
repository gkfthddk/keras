import os
import sys
import subprocess
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.tick_params(labelsize=15)
import random
import ROOT as rt
ptmin=0
ptmax=0
now=datetime.now()
fs=25
for pt in [100,200,500,1000]:
  plt.figure(figsize=(12,8))
  for sample in ["zq","qq","zg","gg"]:
    f=open("{}ptlist{}".format(sample,pt),"read")
    data=[0]+eval(f.readline())
    if("zq" == sample):
      plt.plot(np.arange(0,65),data,label=r"pp$\rightarrow$zq",drawstyle='steps',linewidth=3)
    if("zg" == sample):
      plt.plot(np.arange(0,65),data,color='red',label=r"pp$\rightarrow$zg",drawstyle='steps',linewidth=3,linestyle='--')
    if("qq" ==sample):
      #plt.plot(range(65),data,color='C0',alpha=0.5,drawstyle='steps',linewidth=2)
      plt.fill_between(range(0,65),data,alpha=0.6,linewidth=2,facecolor='azure',edgecolor='C0',label=r"pp$\rightarrow$qq",step='pre')

    if("gg" ==sample):
      plt.fill_between(range(0,65),data,facecolor="#ffcfdc",edgecolor='C1',alpha=0.3,linewidth=2,label=r"pp$\rightarrow$gg",step='pre',linestyle='--')
    
  plt.xlabel("Particle sequence",fontsize=fs*.8)
  plt.ylabel("Average $p_T$",fontsize=fs*.8)
  plt.yscale('log')
  plt.grid(alpha=0.6)
  plt.legend(fontsize=fs)
  plt.title("{}~{}GeV".format(pt,int(pt*1.1)),fontsize=fs)
  a1,a2,b1,b2=plt.axis()
  plt.axis((0,64,b1,b2))
  f.close()
  #plt.show()
  plt.savefig("pic/{}ptlist{}".format(sample,pt),bbox_inches='tight',pad_inches=0.5,dpi=300)
  plt.figure(figsize=(12,8))
  for sample in ["zq","qq","zg","gg"]:
    f=open("{}drlist{}".format(sample,pt),"read")
    data=[0]+eval(f.readline())
    if("zq" == sample):
      plt.plot(np.arange(0,65),data,label=r"pp$\rightarrow$zq",drawstyle='steps',linewidth=3)
    if("zg" == sample):
      plt.plot(np.arange(0,65),data,color='red',label=r"pp$\rightarrow$zg",drawstyle='steps',linewidth=3,linestyle='--')
    if("qq" ==sample):
      #plt.plot(range(65),data,color='C0',alpha=0.5,drawstyle='steps',linewidth=2)
      plt.fill_between(range(0,65),data,alpha=0.6,linewidth=2,facecolor='azure',edgecolor='C0',label=r"pp$\rightarrow$qq",step='pre')

    if("gg" ==sample):
      plt.fill_between(range(0,65),data,facecolor="#ffcfdc",edgecolor='C1',alpha=0.3,linewidth=2,label=r"pp$\rightarrow$gg",step='pre',linestyle='--')
    
  plt.xlabel("Particle sequence",fontsize=fs*.8)
  plt.ylabel("Average $\Delta R$",fontsize=fs*.8)
  plt.yscale('log')
  plt.grid(alpha=0.6)
  plt.legend(fontsize=fs)
  plt.title("{}~{}GeV".format(pt,int(pt*1.1)),fontsize=fs)
  a1,a2,b1,b2=plt.axis()
  plt.axis((0,64,b1,b2))
  f.close()
  #plt.show()
  plt.savefig("pic/{}drlist{}".format(sample,pt),bbox_inches='tight',pad_inches=0.5,dpi=300)

