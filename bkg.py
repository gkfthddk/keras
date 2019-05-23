import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
fs=25
plt.figure(figsize=(12, 8))
plt.ylabel("Gluon efficiency(%)",fontsize=fs*1.0)
plt.xlabel("$p_T$ Range(GeV)",fontsize=fs*1.3)
cl=['C2','C0','C1','C3']
nl=['asubdt{}nocut','asuzjcnn{}nocut','asuzjrnn{}nocut']
#nl=['asuzjcnn{}noetaacut','asuzjcnn{}acut','asuzjcnn{}ptacut','asuzjcnn{}ptgausacut']
#nl=['asubdt{}noeta','asubdt{}','asubdt{}pt']
#nl=['asuzjrnn{}noetaacut','asuzjrnn{}acut','asuzjrnn{}ptacut']
#nl=['asuzqcnn{}eta','asuqqcnn{}eta']
#ll=['BDT-','CNN-','RNN-']
#ll=['etacut-','etacut-']
#ll=['nocut-','etacut-','ptcut-','ptguas-']
ll=['BDT-','CNN-','RNN-',]
event=["Z+jet05","dijet05"]
aucs=[]
for j in range(2):
  for i in range(len(nl)):
    aucs.append({"Z+jet05":[],"dijet05":[]})
    if("zq" in nl[i] and j==1):continue
    if("qq" in nl[i] and j==0):continue
    for pt in [100,200,500,1000]:
      print("aucs/"+nl[i].format(pt))
      dic=eval(open("aucs/"+nl[i].format(pt)).readline())
      aucs[i][event[j]].append(dic[event[j]])
    if(event[j]=="Z+jet05"):
      plt.plot([105,210,525,1050],np.array(aucs[i][event[j]])*100,
          ':',linewidth=3,label=ll[i]+event[j][:-2],marker='o',
          alpha=0.7,color=cl[i],markersize=fs)
    if(event[j]=="dijet05"):
      plt.plot([105,210,525,1050],np.array(aucs[i][event[j]])*100,
          ':',linewidth=3,label=ll[i]+event[j][:-2],marker='o',
          fillstyle='none',color=cl[i],markersize=fs)

plt.xticks([105,210,525,1050],["100\n~110","200\n~220","500\n~550","1000\n~1100"],size=fs*0.8)
plt.yticks([5,6,7,8,9,10],size=fs)
plt.grid(alpha=0.6)
plt.legend(fontsize=fs*0.88,ncol=2,)
a1,a2,b1,b2=plt.axis()
plt.axis((a1,a2,5,10))
#plt.show()
plt.savefig("plots/alletabkg.pdf",bbox_inches='tight',pad_inches=0.5,dpi=300)
plt.savefig("plots/alletabkg.png",bbox_inches='tight',pad_inches=0.5,dpi=300)

"""Bd=[100*(1-eval(i)) for i in "0.918 0.931 0.9297 0.9296".split(" ")]
Bz=[100*(1-eval(i)) for i in "0.921 0.935 0.9417 0.9428".split(" ")]
Cd=[100*(1-eval(i)) for i in "0.925 0.918 0.919 0.901".split(" ")]
Cz=[100*(1-eval(i)) for i in "0.926 0.933 0.942 0.92".split(" ")]
Rd=[100*(1-eval(i)) for i in "0.915 0.926 0.916 0.906".split(" ")]
Rz=[100*(1-eval(i)) for i in "0.913 0.935 0.937 0.93".split(" ")]
plt.figure(figsize=(12, 8))
plt.ylabel("Gluon efficiency(%)",fontsize=fs*1.0)
plt.xlabel("$p_T$ Range(GeV)",fontsize=fs*1.3)
plt.plot([105,210,525,1050],Bz,
          ':',linewidth=3,label="BDT-Z+jet",marker='o',
          alpha=0.7,color='C2',markersize=fs)
plt.plot([105,210,525,1050],Cz,
          ':',linewidth=3,label="CNN-Z+jet",marker='D',
          alpha=0.7,color='C0',markersize=fs*0.75)
plt.plot([105,210,525,1050],Rz,
          ':',linewidth=3,label="RNN-Z+jet",marker='^',
          alpha=0.7,color='C1',markersize=fs)
plt.plot([105,210,525,1050],Bd,
          ':',linewidth=3,label="BDT-dijet",marker='o',
          fillstyle='none',color='C2',markersize=fs)
plt.plot([105,210,525,1050],Cd,
          ':',linewidth=3,label="CNN-dijet",marker='D',
          fillstyle='none',color='C0',markersize=fs*0.75)
plt.plot([105,210,525,1050],Rd,
          ':',linewidth=3,label="RNN-dijet",marker='^',
          fillstyle='none',color='C1',markersize=fs)
plt.xticks([105,210,525,1050],["100\n~110","200\n~220","500\n~550","1000\n~1100"],size=fs*0.8)
plt.yticks([100*0.05,100*0.06,100*0.07,100*0.08,100*0.09,100*0.10],size=fs)

plt.grid(alpha=0.6)
plt.legend(fontsize=fs*0.8,loc=2,ncol=2)
plt.savefig("background.pdf",bbox_inches='tight',pad_inches=0.5,dpi=300)
plt.savefig("background.png",bbox_inches='tight',pad_inches=0.5,dpi=300)
"""
