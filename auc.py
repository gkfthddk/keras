import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
fs=25
plt.figure(figsize=(12, 8))
plt.ylabel("ROC AUC",fontsize=fs*1.3)
plt.xlabel("$p_T$ Range(GeV)",fontsize=fs*1.3)
cl=['C2','C0','C1','C3']
nl=['asubdt{}nocut','asuzjcnn{}nocut','asuzjrnn{}nocut']
#nl=['asuzjcnn{}noetaacut','asuzjcnn{}acut','asuzjcnn{}ptacut',]
#nl=['asubdt{}noeta','asubdt{}','asubdt{}pt']
#nl=['asuzjrnn{}noetaacut','asuzjrnn{}acut','asuzjrnn{}ptacut']
#nl=['asuzqcnn{}noetaetacut','asuqqcnn{}noetaetacut','asuzqcnn{}eta','asuqqcnn{}eta']
#ll=['BDT-','CNN-','RNN-']
#ll=['nocut-','nocut-','etacut-','etacut-']
ll=['BDT-','CNN-','RNN-',]
event=["Z+jet","dijet"]
aucs=[]
for j in range(2):
  for i in range(len(nl)):
    aucs.append({"Z+jet":[],"dijet":[]})
    if("zq" in nl[i] and j==1):continue
    if("qq" in nl[i] and j==0):continue
    for pt in [100,200,500,1000]:
      print("aucs/"+nl[i].format(pt))
      dic=eval(open("aucs/"+nl[i].format(pt)).readline())
      aucs[i][event[j]].append(dic[event[j]])
    if(event[j]=="Z+jet"):
      plt.plot([105,210,525,1050],aucs[i][event[j]],
          ':',linewidth=3,label=ll[i]+event[j],marker='o',
          alpha=0.7,color=cl[i],markersize=fs)
    if(event[j]=="dijet"):
      plt.plot([105,210,525,1050],aucs[i][event[j]],
          ':',linewidth=3,label=ll[i]+event[j],marker='o',
          fillstyle='none',color=cl[i],markersize=fs)

plt.xticks([105,210,525,1050],["100\n~110","200\n~220","500\n~550","1000\n~1100"],size=fs*0.8)
plt.yticks([0.81,0.82,0.83,0.84,0.85,0.86,0.87],size=fs)
plt.grid(alpha=0.6)
plt.legend(fontsize=fs*0.88,ncol=2,loc=8)
a1,a2,b1,b2=plt.axis()
plt.axis((a1,a2,0.81,0.87))
#plt.title("cnnetacut")
#plt.show()
plt.savefig("plots/alletacut.pdf",bbox_inches='tight',pad_inches=0.5,dpi=300)
plt.savefig("plots/alletacut.png",bbox_inches='tight',pad_inches=0.5,dpi=300)
"""plt.plot([105,210,525,1050],[0.815,0.833,0.846,0.841],
          ':',linewidth=3,label="BDT-Z+jet",marker='o',
          alpha=0.7,color='C2',markersize=fs)
plt.plot([105,210,525,1050],[0.831, 0.856, 0.87, 0.842],
          ':',linewidth=3,label="CNN-Z+jet",marker='D',
          alpha=0.7,color='C0',markersize=fs*0.75)
plt.plot([105,210,525,1050],[0.814,0.835,0.835,0.805],
          ':',linewidth=3,label="RNN-Z+jet",marker='^',
          alpha=0.7,color='C1',markersize=fs)
plt.plot([105,210,525,1050],[0.811,0.827,0.832,0.825],
          ':',linewidth=3,label="BDT-Dijet",marker='o',
          fillstyle='none',color='C2',markersize=fs)
plt.plot([105,210,525,1050],[0.827, 0.844, 0.846, 0.824],
          ':',linewidth=3,label="CNN-Dijet",marker='D',
          fillstyle='none',color='C0',markersize=fs*0.75)
plt.plot([105,210,525,1050],[0.825,0.822,0.798,0.775],
          ':',linewidth=3,label="RNN-Dijet",marker='^',
          fillstyle='none',color='C1',markersize=fs)
plt.xticks([105,210,525,1050],["100\n~110","200\n~220","500\n~550","1000\n~1100"],size=fs*0.8)
plt.yticks([0.76,0.78,0.8,0.82,0.84,0.86],size=fs)
plt.grid(alpha=0.6)
plt.legend(fontsize=fs*0.8,ncol=2)
plt.show()
#plt.savefig("realistic.pdf",bbox_inches='tight',pad_inches=0.5,dpi=300)
#plt.savefig("realistic.png",bbox_inches='tight',pad_inches=0.5,dpi=300)
plt.figure(figsize=(12, 8))
plt.ylabel("ROC AUC",fontsize=fs*1.3)
plt.xlabel("$p_T$ Range(GeV)",fontsize=fs*1.3)
plt.plot([105,210,525,1050],[0.85,0.87,0.889,0.894],
          ':',linewidth=3,label="CNN-pure",marker='D',
          color='C0',fillstyle='none',markersize=fs*0.75)
plt.plot([105,210,525,1050],[0.845,0.86,0.87,0.868],
          ':',linewidth=3,label="RNN-pure",marker='^',
          color='C1',fillstyle='none',markersize=fs)
plt.plot([105,210,525,1050],[0.831,0.849,0.853,0.822],
          ':',linewidth=3,label="CNN-realistic",marker='D',
          alpha=0.7,color='C0',markersize=fs*0.75)
plt.plot([105,210,525,1050],[0.814,0.835,0.835,0.805],
          ':',linewidth=3,label="RNN-realistic",marker='^',
          alpha=0.7,color='C1',markersize=fs)
plt.xticks([105,210,525,1050],["100\n~110","200\n~220","500\n~550","1000\n~1100"],size=fs*0.8)
plt.yticks([0.8,0.82,0.84,0.86,0.88,0.9],size=fs)
plt.grid(alpha=0.6)
plt.legend(fontsize=fs*0.8,ncol=2,loc=8)
plt.savefig("pure.pdf",bbox_inches='tight',pad_inches=0.5,dpi=300)
plt.savefig("pure.png",bbox_inches='tight',pad_inches=0.5,dpi=300)
"""
