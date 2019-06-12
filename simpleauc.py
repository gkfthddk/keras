import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#nl=['asubdt{}ptonlyptcut','asuzjrnn{}ptonlyptcut','asuzjrnn{}pteta31ptcut']
nl=['asubdt{}ptonlyptcut','asuzjcnn{}ptonlyptcut','asuzjcnn{}ptonly2ptcut']
#nl2=['asubdt{}ptetaptcut','asuzjrnn{}ptonlyadamptcut','asuzjrnn{}ptetaadamptcut']
nl2=['asubdt{}ptetaptcut','asuzjrnn{}ptptcut','asuzjrnn{}pteta21ptcut']

event=["Z+jet","dijet","Z+jet05","dijet05"]
aucs=[]
for j in range(2):
  for i in range(len(nl)):
    if("zq" in nl[i] and j==1):continue
    if("qq" in nl[i] and j==0):continue
    aucs.append({"Z+jet":[],"dijet":[],"Z+jet05":[],"dijet05":[]})
    for pt in [100,200,500,1000]:
      print("aucs/"+nl[i].format(pt))
      if(j==1):dic=eval(open("aucs/"+nl2[i].format(pt)).readline())
      if(j==0):dic=eval(open("aucs/"+nl[i].format(pt)).readline())
      print(pt,dic)
      for k in range(len(event)):
        aucs[i+j*3][event[k]].append(dic[event[k]])

fs=25
plt.figure(figsize=(12, 8))
plt.ylabel("ROC AUC",fontsize=fs*1.3)
plt.xlabel("$p_T$ Range(GeV)",fontsize=fs*1.3)
cl=['C2','C0','C1','C3']
#nl=['asubdt{}ptonlyptcut','asuzjcnn{}ptonlyptcut','asubdt{}ptptcut','asuzjcnn{}ptptcut',]#'asuzjrnn{}ptonlypt']
#nl=['asuzjrnn{}ptptcut','asuzjrnn{}ptonly31ptcut','asuzjrnn{}ptonlyptcut']
#nl=['asubdt{}ptnocut','asuzjcnn{}ptnocut','asuzjrnn{}ptnocut']
#nl=['asubdt{}noeta','asubdt{}','asubdt{}pt']
#nl=['asuzjrnn{}noetaacut','asuzjrnn{}acut','asuzjrnn{}ptacut']
#nl=['asuzqcnn{}noetaetacut','asuqqcnn{}noetaetacut','asuzqcnn{}eta','asuqqcnn{}eta']
#ll=['$p_T$ cut-','$\eta,p_T$ cut-','$p_T$ cut-','etacut-']
ll=['BDT-','CNN-','RNN-',]
#ll=['BDT-ptcut-','CNN-ptcut-','BDT-ptetacut-','CNN-ptetacut-','RNN-',]
#ll=['pt-','31-','21-','331-']
mak=["o","D","^","o","o"]
ms=[1,0.75,1,1,1]

for j in range(2):
  for i in range(len(nl)):
    if("zq" in nl[i] and j==1):continue
    if("qq" in nl[i] and j==0):continue
    if(event[j]=="Z+jet"):
      plt.plot([105,210,525,1050],aucs[i][event[j]],
          ':',linewidth=3,label=ll[i]+event[j],marker=mak[i],
          alpha=0.7,color=cl[i],markersize=fs*ms[i])
    if(event[j]=="dijet"):
      plt.plot([105,210,525,1050],aucs[i][event[j]],
          ':',linewidth=3,label=ll[i]+event[j],marker=mak[i],
          fillstyle='none',color=cl[i],markersize=fs*ms[i])

plt.xticks([105,210,525,1050],["100\n~110","200\n~220","500\n~550","1000\n~1100"],size=fs*0.8)
plt.yticks([0.8,0.81,0.82,0.83,0.84,0.85,0.86,0.87],size=fs)
plt.grid(alpha=0.6)
plt.legend(fontsize=fs*0.88,ncol=2,loc=8)
#a1,a2,b1,b2=plt.axis()
#plt.axis((a1,a2,0.81,0.87))
#plt.title("cnnetacut")
#plt.show()
plt.savefig("plots/allptcut.pdf",bbox_inches='tight',pad_inches=0.5,dpi=300)
plt.savefig("plots/allptcut.png",bbox_inches='tight',pad_inches=0.5,dpi=300)


plt.show()

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
