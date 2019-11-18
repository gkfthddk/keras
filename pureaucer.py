import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#nl=['asubdt{}ptetaptcut','asuzjrnn{}pteta21ptcut','asuzjcnn{}ptptcut']
#nl2=['asubdt{}ptetaptcut','asuzjrnn{}pteta21ptcut','asuzjcnn{}ptptcut']
nl=['asuzqbdt{}ptptcut','asuzqrnn{}ptonly21ptcut','asuzqcnn{}ptonlyptcut']
nl2=['asuqqbdt{}ptptcut','asuqqrnn{}ptonly21ptcut','asuqqcnn{}ptonlyptcut']
#nl=['asubdt{}ptonlyptcut','asuzjrnn{}ptonly21ptcut','asuzjcnn{}ptonly3ptcut']
#nl2=['asubdt{}ptonlyptcut','asuzjrnn{}ptonly21ptcut','asuzjcnn{}ptonly3ptcut']

#nl2=['npzbdt{}ptptcut','npzcnn{}ptptcut','npznnn{}ptptcut','npzrnn21{}ptptcut']
#nl=['npzbdt{}ptetaptcut','npzcnn{}ptetaptcut','npznnn{}ptetaptcut','npzrnn21{}ptetaptcut']
#nl=['npzbdt{}ptptcut','npzcnn{}ptptcut','npzrnn21{}ptptcut']
#nl2=['npzbdt{}ptetaptcut','npzcnn{}ptetaptcut','npzrnn21{}ptetaptcut']

save="asupurept"

event=["Z+jet","dijet","Z+jet05","dijet05"]
aucs=[]
for j in range(2):
  for i in range(len(nl)):
    if("npz" in nl[i]):
      if("npzzq" in nl[i] and j==1):continue
      if("npzqq" in nl[i] and j==0):continue
    else:
      pass
      #if("zq" in nl[i] and j==1):continue
      #if("qq" in nl[i] and j==0):continue
    aucs.append({"Z+jet":[],"dijet":[],"Z+jet05":[],"dijet05":[]})
    for pt in [100,200,500,1000]:
      print("aucs/"+nl[i].format(pt))
      if(j==1):
        dic=eval(open("aucs/"+nl2[i].format(pt)).readline())
        ev="dijet"
      if(j==0):
        dic=eval(open("aucs/"+nl[i].format(pt)).readline())
        ev="Z+jet"
      print(pt,dic)
      aucs[i+j*3][ev].append(dic[ev])
print(aucs)
fs=25
plt.figure(figsize=(12, 8))
plt.ylabel("ROC AUC",fontsize=fs*1.3)
plt.xlabel("$p_T$ Range(GeV)",fontsize=fs*1.3)
cl=['C2','C1','C0']
#nl=['asubdt{}ptonlyptcut','asuzjcnn{}ptonlyptcut','asubdt{}ptptcut','asuzjcnn{}ptptcut',]#'asuzjrnn{}ptonlypt']
#nl=['asuzjrnn{}ptptcut','asuzjrnn{}ptonly31ptcut','asuzjrnn{}ptonlyptcut']
#nl=['asubdt{}ptnocut','asuzjcnn{}ptnocut','asuzjrnn{}ptnocut']
#nl=['asubdt{}noeta','asubdt{}','asubdt{}pt']
#nl=['asuzjrnn{}noetaacut','asuzjrnn{}acut','asuzjrnn{}ptacut']
#nl=['asuzqcnn{}noetaetacut','asuqqcnn{}noetaetacut','asuzqcnn{}eta','asuqqcnn{}eta']
#ll=['$p_T$ cut-','$\eta,p_T$ cut-','$p_T$ cut-','etacut-']
ll=['BDT-','RNN-','CNN-']
#ll=['BDT-ptcut-','CNN-ptcut-','BDT-ptetacut-','CNN-ptetacut-','RNN-',]
#ll=['pt-','31-','21-','331-']
mak=["D","^","o","v"]
ms=[0.75,1,1,1,1]

for j in range(2):
  for i in range(len(nl)):
    if("npz" in nl[i]):
      if("npzzq" in nl[i] and j==1):continue
      if("npzqq" in nl[i] and j==0):continue
    else:
      pass
    if(event[j]=="Z+jet"):
      print("222")
      plt.plot([105,210,525,1050],aucs[i][event[j]],
          ':',linewidth=3,label=ll[i]+event[j],marker=mak[i],
          alpha=0.7,color=cl[i],markersize=fs*ms[i])
    if(event[j]=="dijet"):
      print("111")
      plt.plot([105,210,525,1050],aucs[i+j*3][event[j]],
          ':',linewidth=3,label=ll[i]+event[j],marker=mak[i],
          fillstyle='none',color=cl[i],markersize=fs*ms[i])

plt.xticks([105,210,525,1050],["100\n~110","200\n~220","500\n~550","1000\n~1100"],size=fs*0.8)
plt.yticks([0.78,0.8,0.82,0.84,0.86,0.88],size=fs)
plt.grid(alpha=0.6)
plt.legend(fontsize=fs*0.88,ncol=2,loc=8)
#a1,a2,b1,b2=plt.axis()
#plt.axis((a1,a2,0.81,0.87))
#plt.title("cnnetacut")
#plt.show()
plt.savefig("plots/{}cut.pdf".format(save),bbox_inches='tight',pad_inches=0.5,dpi=300)
plt.savefig("plots/{}cut.png".format(save),bbox_inches='tight',pad_inches=0.5,dpi=300)


