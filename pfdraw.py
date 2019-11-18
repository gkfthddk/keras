import ROOT as rt
import os
import sys
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
fs=25
plt.rc('xtick',labelsize=int(fs*.8))
plt.rc('ytick',labelsize=int(fs*.8))
import random
ptmin=0
ptmax=0
now=datetime.now()
for pt in [500]:
  #f=np.load("gendr{}.npz".format(pt))
  f=np.load("jj25dr{}.npz".format(pt))
  seq=f["seqset"]
  eve=f["eveset"]
  pairlist=f["pairlist"]
  num=64
  q1=np.zeros(num)
  q2=np.zeros(num)
  g1=np.zeros(num)
  g2=np.zeros(num)
  nq1=0
  nq2=0
  ng1=0
  ng2=0
  for i in range(len(eve)):
    pair=pairlist[np.argmax(eve[i])]
    if(pair[0]=="q"):
      #q1+=seq[0][i][:,0]
      q1+=np.sqrt(pow(seq[0][i][:,1],2)+pow(seq[0][i][:,1],2))
      nq1+=1
    if(pair[1]=="q"):
      #q2+=seq[1][i][:,0]
      q1+=np.sqrt(pow(seq[1][i][:,1],2)+pow(seq[1][i][:,1],2))
      nq1+=1
    if(pair[0]=="g"):
      #g1+=seq[0][i][:,0]
      g1+=np.sqrt(pow(seq[0][i][:,1],2)+pow(seq[0][i][:,1],2))
      ng1+=1
    if(pair[1]=="g"):
      #g2+=seq[1][i][:,0]
      g1+=np.sqrt(pow(seq[1][i][:,1],2)+pow(seq[1][i][:,1],2))
      ng1+=1
  q1=q1/nq1
  #q2=q2/nq2
  g1=g1/ng1
  #g2=g2/ng2
  q1=np.concatenate([[0],q1])
  q2=np.concatenate([[0],q2])
  g1=np.concatenate([[0],g1])
  g2=np.concatenate([[0],g2])
  plt.figure(figsize=(12,8))
  plt.xlabel("Order of particle in sequence",fontsize=fs*1.)
  plt.ylabel("Average $\Delta R$",fontsize=fs*1.3)
  plt.fill_between(np.arange(0,num+1),q1,alpha=0.6,linewidth=2,facecolor='azure',edgecolor='C0',label=r"Quark jet",step='pre')
  plt.plot(np.arange(0,num+1),q1,color='C0',alpha=0.6,drawstyle='steps',linewidth=2)
  plt.fill_between(np.arange(0,num+1),g1,facecolor="#ffcfdc",edgecolor='red',alpha=0.4,linewidth=2,label=r"Gluon jet",step='pre',linestyle='-')
  #plt.plot(np.arange(0,num+1),q2,label=r"pp$\rightarrow$zq",drawstyle='steps',linewidth=3)
  #plt.plot(np.arange(0,num+1),g2,color='red',label=r"pp$\rightarrow$zg",drawstyle='steps',linewidth=3,linestyle='--')
  #  if("qq" ==sample):
  #    #plt.plot(range(65),data,color='C0',alpha=0.5,drawstyle='steps',linewidth=2)
  #    plt.fill_between(range(0,65),data,alpha=0.6,linewidth=2,facecolor='azure',edgecolor='C0',label=r"pp$\rightarrow$qq",step='pre')

  #  if("gg" ==sample):
  #    plt.fill_between(range(0,65),data,facecolor="#ffcfdc",edgecolor='C1',alpha=0.3,linewidth=2,label=r"pp$\rightarrow$gg",step='pre',linestyle='--')
  #plt.ylabel("Average $p_T$",fontsize=fs*.8)
  #plt.yscale('log')
  plt.grid(alpha=0.6)
  plt.legend(fontsize=fs)
  plt.title("jet $p_T$ range {}~{} GeV".format(pt,int(pt*1.1)),fontdict={"weight":"bold","size":fs*1.})
  a1,a2,b1,b2=plt.axis()
  plt.axis((0,64,b1,b2))
  f.close()
  plt.savefig("pic/pfdrlist{}".format(pt),bbox_inches='tight',pad_inches=0.5,dpi=300)
  q1=np.zeros(num)
  q2=np.zeros(num)
  g1=np.zeros(num)
  g2=np.zeros(num)
  nq1=0
  nq2=0
  ng1=0
  ng2=0
  for i in range(len(eve)):
    pair=pairlist[np.argmax(eve[i])]
    if(pair[0]=="q"):
      q1+=seq[0][i][:,0]
      nq1+=1
    if(pair[1]=="q"):
      q1+=seq[1][i][:,0]
      nq1+=1
    if(pair[0]=="g"):
      g1+=seq[0][i][:,0]
      ng1+=1
    if(pair[1]=="g"):
      g1+=seq[1][i][:,0]
      ng1+=1
  q1=q1/nq1
  #q2=q2/nq2
  g1=g1/ng1
  #g2=g2/ng2
  q1=np.concatenate([[0],q1])
  q2=np.concatenate([[0],q2])
  g1=np.concatenate([[0],g1])
  g2=np.concatenate([[0],g2])
  plt.figure(figsize=(12,8))
  plt.fill_between(np.arange(0,num+1),q1,alpha=0.6,linewidth=2,facecolor='azure',edgecolor='C0',label=r"Quark jet",step='pre')
  plt.plot(np.arange(0,num+1),q1,color='C0',alpha=0.6,drawstyle='steps',linewidth=2)
  plt.fill_between(np.arange(0,num+1),g1,facecolor="#ffcfdc",edgecolor='red',alpha=0.4,linewidth=2,label=r"Gluon jet",step='pre',linestyle='-')
  #plt.plot(np.arange(0,num+1),q2,label=r"pp$\rightarrow$zq",drawstyle='steps',linewidth=3)
  #plt.plot(np.arange(0,num+1),g2,color='red',label=r"pp$\rightarrow$zg",drawstyle='steps',linewidth=3,linestyle='--')
  #  if("qq" ==sample):
  #    #plt.plot(range(65),data,color='C0',alpha=0.5,drawstyle='steps',linewidth=2)
  #    plt.fill_between(range(0,65),data,alpha=0.6,linewidth=2,facecolor='azure',edgecolor='C0',label=r"pp$\rightarrow$qq",step='pre')

  #  if("gg" ==sample):
  #    plt.fill_between(range(0,65),data,facecolor="#ffcfdc",edgecolor='C1',alpha=0.3,linewidth=2,label=r"pp$\rightarrow$gg",step='pre',linestyle='--')
  plt.xlabel("Order of particle in sequence",fontsize=fs*1.)
  plt.ylabel("Average $p_T$",fontsize=fs*1.3)
  #plt.ylabel("Average $\Delta R$",fontsize=fs*.8)
  plt.yscale('log')
  plt.grid(alpha=0.6)
  plt.legend(fontsize=fs)
  plt.title("jet $p_T$ range {}~{} GeV".format(pt,int(pt*1.1)),fontdict={"weight":"bold","size":fs*1.})
  a1,a2,b1,b2=plt.axis()
  plt.axis((0,64,b1,b2))
  f.close()
  plt.savefig("pic/pfptlist{}".format(pt),bbox_inches='tight',pad_inches=0.5,dpi=300)
plt.show()
