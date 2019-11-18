import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--name",type=str,default="asubdt{}ptetaptcut,asuzjcnn{}ptptcut,asuzjrnn{}pteta21ptcut",help="")
args=parser.parse_args()

nl=args.name.split(",")
save="allpteta"

event=["Z+jet","dijet","Z+jet05","dijet05"]
aucs=[]
for i in range(len(nl)):
  if("zq" in nl[i] and j==1):continue
  if("qq" in nl[i] and j==0):continue
  aucs.append({"Z+jet":[],"dijet":[],"Z+jet05":[],"dijet05":[]})
  for pt in [100,200,500,1000]:
    print("aucs/"+nl[i].format(pt))
    dic=eval(open("aucs/"+nl[i].format(pt)).readline())
    print(pt,dic)
    for k in range(len(event)):
      aucs[i][event[k]].append(dic[event[k]])

fs=25
plt.figure(figsize=(12, 8))
plt.ylabel("ROC AUC",fontsize=fs*1.3)
plt.xlabel("$p_T$ Range(GeV)",fontsize=fs*1.3)
cl=['C2','C0','C1','C3','C4','C5']
#ll=['BDT-','CNN-','RNN-',]
ll=nl
mak=["o","o","o","o","o","o"]

for j in range(2):
  for i in range(len(nl)):
    if("zq" in nl[i] and j==1):continue
    if("qq" in nl[i] and j==0):continue
    if(event[j]=="Z+jet"):
      plt.plot([105,210,525,1050],aucs[i][event[j]],
          ':',linewidth=3,label=ll[i]+event[j],marker=mak[i],
          alpha=0.7,color=cl[i],markersize=fs)
    if(event[j]=="dijet"):
      plt.plot([105,210,525,1050],aucs[i][event[j]],
          ':',linewidth=3,label=ll[i]+event[j],marker=mak[i],
          fillstyle='none',color=cl[i],markersize=fs)
plt.legend(fontsize=fs*0.88,ncol=2,loc=8)
plt.xticks([105,210,525,1050],["100\n~110","200\n~220","500\n~550","1000\n~1100"],size=fs*0.8)
plt.yticks([0.78,0.8,0.82,0.84,0.86,0.88],size=fs)
plt.grid(alpha=0.6)
plt.show()
