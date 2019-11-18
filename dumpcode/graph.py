import numpy as np
import datetime
import random
import ROOT as rt
from ROOT import gPad,gStyle
import math
import sys
from array import array
import matplotlib.pyplot as plt
from scipy.interpolate import splrep,splev

def smooth(a):
  y=np.array(a)
  x=np.arange(1,len(y)+1)
  print x
  tck=splrep(x,y)
  xnew=np.linspace(x[0],x[-1],40)
  ynew=splev(xnew,tck)
  return [xnew,ynew]

arnum=16
qpt=rt.TH1F("qpt","quark",20,0,20)
pt="100"
f=open("save/ten"+pt+"simple3/history")
a=eval(f.readlines()[1])['acc']
f.close()
f=open("save/ten"+pt+"clstm4/history")
b=eval(f.readlines()[1])['acc']
f.close()
f=open("save/ten"+pt+"simplecnn3/history")
c=eval(f.readlines()[1])['acc']
f.close()
f=open("save/ten"+pt+"simple3/history")
aa=eval(f.readlines()[1])['val_acc']
f.close()
f=open("save/ten"+pt+"clstm4/history")
bb=eval(f.readlines()[1])['val_acc']
f.close()
f=open("save/ten"+pt+"simplecnn3/history")
cc=eval(f.readlines()[1])['val_acc']
f.close()
arnum=16
xnew,ynew=smooth(a)
x=range(len(a))
#plt.plot(x,a,x,b,x,c,x,d,x,e)
plt.plot(x,a,"--",label="RNN+CNN-train")
plt.plot(x,aa,label="RNN+CNN-val")
plt.plot(x,c,"--",label="CNN-train")
plt.plot(x,cc,label="CNN-val")
plt.plot(x,b,"--",label="RNN-train")
plt.plot(x,bb,label="RNN-val")
plt.title(pt)
plt.xticks(range(0,20,3))
plt.xlabel("Training step")
plt.ylabel("Accuracy")
a1,a2,b1,b2=plt.axis()
plt.axis([a1,a2,0.7,0.8])
#plt.plot(range(10),a,"b",range(10),b,"r")
plt.legend()
plt.show()

