import numpy as np
import matplotlib
import matplotlib.pyplot as plt

nl=['asuzqcnn{}ptonlyptcut','asuqqcnn{}ptonlyptcut','asuzjcnn{}ptonly3ptcut','asuzjcnn{}ptonly3ptcut']
#nl=['asuzqcnn{}ptetaptcut','asuqqcnn{}ptetaptcut','asuzjcnn{}ptptcut','asuzjcnn{}ptptcut']
#nl=['npzzqcnn{}ptetaptcut','npzqqcnn{}ptetaptcut','npzzqbdt{}ptetaptcut','npzqqbdt{}ptetaptcut']

name="asupurept"

ne=["Z+jet","dijet","Z+jet","dijet"]
#event=["Z+jet"]
event=["Z+jet","dijet","Z+jet05","dijet05"]
ll=['pure-','pure-','realistic-','realistic-']
aucs=[]
for j in range(0,2):
  for i in range(len(nl)):
    aucs.append({"Z+jet":[],"dijet":[],"Z+jet05":[],"dijet05":[]})
    if("npz" in nl[i]):
      if("npzzq" in nl[i] and j==1):continue
      if("npzqq" in nl[i] and j==0):continue
    else:
      if("zq" in nl[i] and j==1):continue
      if("qq" in nl[i] and j==0):continue
    for pt in [100,200,500,1000]:
      print("aucs/"+nl[i].format(pt))
      dic=eval(open("aucs/"+nl[i].format(pt)).readline())
      print(pt,dic,i,j)
      aucs[i][event[j]].append(dic[event[j]])

fs=25
plt.figure(figsize=(12, 8))
plt.ylabel("ROC AUC",fontsize=fs*1.3)
plt.xlabel("$p_T$ Range (GeV)",fontsize=fs*1.3)
cl=['C0','C0','C1','C1']
#ll=['BDT-ptcut-','CNN-ptcut-','BDT-ptetacut-','CNN-ptetacut-','RNN-',]
#ll=['pt-','31-','21-','331-']
mak=["D","D","^","^","o"]
ms=[0.75,0.75,1,1,1]

for i in range(len(nl)):
  if(i%2==0):
    plt.plot([105,210,525,1050],aucs[i][ne[i]],
        ':',linewidth=3,label=ll[i]+ne[i],marker=mak[i],
        alpha=0.7,color=cl[i],markersize=fs*ms[i])
  if(i%2==1):
    plt.plot([105,210,525,1050],aucs[i][ne[i]],
        ':',linewidth=3,label=ll[i]+ne[i],marker=mak[i],
        fillstyle='none',color=cl[i],markersize=fs*ms[i])

plt.xticks([105,210,525,1050],["100\n~120","200\n~240","500\n~600","1000\n~1200"],size=fs*0.8)
plt.yticks([0.80,0.82,0.84,0.86,0.88,0.90],size=fs)
plt.grid(alpha=0.6)
plt.legend(fontsize=fs*0.88,ncol=2,loc=8)
#a1,a2,b1,b2=plt.axis()
#plt.axis((a1,a2,0.81,0.87))
#plt.title("cnnetacut")
plt.savefig("plots/{}cut.pdf".format(name),bbox_inches='tight',pad_inches=0.5,dpi=300)
plt.savefig("plots/{}cut.png".format(name),bbox_inches='tight',pad_inches=0.5,dpi=300)
plt.show()

