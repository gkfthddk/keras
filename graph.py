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
  x=np.arange(len(y))
  print x
  tck=splrep(x,y)
  xnew=np.linspace(x[0],x[-1],50)
  ynew=splev(xnew,tck)
  return [xnew,ynew]

arnum=16
qpt=rt.TH1F("qpt","quark",20,0,20)
f=open("save/qg100gruasym/history")
a=eval(f.readlines()[1])['acc']
f.close()
f=open("save/qg100rnn/history")
b=eval(f.readlines()[1])['acc']
f.close()
f=open("save/qg100asym/history")
c=eval(f.readlines()[1])['acc']
f.close()
f=open("save/qg100simple/history")
d=eval(f.readlines()[1])['acc']
f.close()
f=open("save/qg100gruasym/history")
aa=eval(f.readlines()[1])['val_acc']
f.close()
f=open("save/qg100rnn/history")
bb=eval(f.readlines()[1])['val_acc']
f.close()
f=open("save/qg100asym/history")
cc=eval(f.readlines()[1])['val_acc']
f.close()
f=open("save/qg100simple/history")
dd=eval(f.readlines()[1])['val_acc']
f.close()
arnum=16
xnew,ynew=smooth(a)
x=range(len(a))
#plt.plot(x,a,x,b,x,c,x,d,x,e)
plt.plot(x,a,"--",label="GRU+CNN-train")
plt.plot(x,aa,label="GRU+CNN-val")
plt.plot(range(20),d,"--",label="Simple-train")
plt.plot(range(20),dd,label="Simple-val")
plt.plot(x,c,"--",label="CNN-train")
plt.plot(x,cc,label="CNN-val")
plt.plot(x,b,"--",label="GRU*-train")
plt.plot(x,bb,label="GRU*-val")
#plt.plot(range(10),a,"b",range(10),b,"r")
plt.legend()
plt.show()

