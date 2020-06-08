import os
import sys
import subprocess
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
#plt.tick_params(labelsize=15)
import random
import ROOT as rt
ptmin=0
ptmax=0
now=datetime.now()
for sample in ["jj"]:
  l=0
  fig,B=plt.subplots(ncols=4,figsize=(16,4))
  if("g" in sample):
    totitle="Gluon charged multiplicity"
  if("q" in sample):
    totitle="Quark charged multiplicity"
  #fig.suptitle(totitle,fontsize=15)
  for pt in [200,500]:
    """f=open("{}chpt{}".format(sample,pt),"read")
    chpt=eval(f.readline())
    #chpt=np.array(chpt)/(1.*max(chpt))
    chpt=np.array(chpt).reshape((33,33))
    B[l].imshow(chpt,extent=[-0.4,0.4,-0.4,0.4])
    #plt.savefig("pic/{}chpt{}".format(sample,pt))
    f.close()"""
    f=open("{}chmt{}".format(sample,pt),"read")
    chmt=eval(f.readline())
    #chmt=np.array(chmt)/(1.*max(chmt))
    chmt=np.array(chmt).reshape((33,33))
    B[l].imshow(chmt,extent=[-0.4,0.4,-0.4,0.4])
    B[l].title.set_text("{}~{}GeV".format(pt,int(pt*1.1)))#,fontsize=15)
    B[l].set_xlabel("$\Delta\eta$",fontsize=11)
    B[l].set_ylabel("$\Delta\phi$",fontsize=11)
    B[l].set_xticks([-0.4,-0.2,0,0.2,0.4])
    B[l].set_yticks([-0.4,-0.2,0,0.2,0.4])#,size=13)
    #plt.savefig("pic/{}chmt{}".format(sample,pt))
    f.close()
    l+=1
  plt.subplots_adjust(top=0.8,left=0.01,right=0.99,wspace=0.05)
  plt.savefig("pic/four{}chmt{}".format(sample,pt))
  plt.show()

