import numpy as np
import random
import ROOT as rt
import math
from siter import *
from array import array
import matplotlib.pyplot as plt
pt=500
tdata="sdata/dijet_{0}_{1}/dijet_{0}_{1}_training.root".format(pt,int(pt*1.1))
train=wkiter([tdata,tdata],batch_size=1024,end=1,rc="rc")
gen=train.next()
fig=plt.figure()
NN=3
gen.next()
[data,label]=gen.next()
#a=np.swapaxes(np.swapaxes(np.array(qim).reshape(3,33,33),0,1),1,2)
#b=np.swapaxes(np.swapaxes(np.array(gim).reshape(3,33,33),0,1),1,2)
i=1
j=1
while True:
  if(i>3 and j>3):break
  print(i)
  num=random.randrange(1,1024)
  #num=i
  img=np.swapaxes(np.swapaxes(data[1][num],0,1),1,2)
  if(label[num][0]==0 and i<=3):
    fig.add_subplot(2,NN,i)
    plt.imshow(img)
    i+=1
  if(label[num][0]==1 and j<=3):
    fig.add_subplot(2,NN,j+NN)
    plt.imshow(img)
    j+=1
"""img=np.swapaxes(np.swapaxes(data[1][0]/10000,0,1),1,2)
while True:
  if(i>10000):break
  for j in range(1,1024):
    if(i>10000):break
    #num=i
    if(label[j][0]==0):
      img+=np.swapaxes(np.swapaxes(data[1][j]/10000,0,1),1,2)
      i+=1
  [data,label]=gen.next()
fig.add_subplot(1,1,1)"""
#fig.add_subplot(2,NN,2)
plt.imshow(img)
plt.show()
"""def qdraw(num):
  qjet.GetEntry(num)
  im=np.swapaxes(np.swapaxes(np.array(qim).reshape(3,33,33),0,1),1,2)
  plt.imshow(im)
  plt.show()
def gdraw(num):
  gjet.GetEntry(num)
  im=np.swapaxes(np.swapaxes(np.array(gim).reshape(3,33,33),0,1),1,2)
  plt.imshow(im)
  plt.show()"""
