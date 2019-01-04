import numpy as np
import random
import ROOT as rt
import math
from array import array
import matplotlib.pyplot as plt
#dataname1="data/ppqq_img.root"
dataname1="../jetdata/ppzj100__img.root"
dataname2=dataname1
#dataname2="data/ppgg_img.root"
qfile=rt.TFile(dataname1,'read')
gfile=rt.TFile(dataname2,'read')
qjet=qfile.Get("image")
gjet=gfile.Get("image")
qim = array('B', [0]*(3*(33)*(33)))
gim = array('B', [0]*(3*(33)*(33)))
qjet.SetBranchAddress("image", qim)
gjet.SetBranchAddress("image", gim)
qlabel = array('B', [0])
glabel = array('B', [0])
qjet.SetBranchAddress("label", qlabel)
gjet.SetBranchAddress("label", glabel)
qEntries=qjet.GetEntriesFast()
gEntries=gjet.GetEntriesFast()

fig=plt.figure()
NN=7
#a=np.swapaxes(np.swapaxes(np.array(qim).reshape(3,33,33),0,1),1,2)
#b=np.swapaxes(np.swapaxes(np.array(gim).reshape(3,33,33),0,1),1,2)
a=(np.array(qim))
b=(np.array(gim))
for i in range(1,NN):
  num=random.randrange(1,10000)
  #num=i
  qjet.GetEntry(num)
  img=np.swapaxes(np.swapaxes(np.array(qim).reshape(3,33,33),0,1),1,2)
  #a+=qim
  fig.add_subplot(2,NN,i)
  plt.imshow(img)
  gjet.GetEntry(num)
  img=np.swapaxes(np.swapaxes(np.array(gim).reshape(3,33,33),0,1),1,2)
  #b+=gim
  fig.add_subplot(2,NN,i+NN)
  plt.imshow(img)
#fig.add_subplot(2,NN,NN)
#fig.add_subplot(2,1,1)
#plt.imshow(np.swapaxes(np.swapaxes(a.reshape(3,33,33),0,1),1,2)/NN*10)
#fig.add_subplot(2,NN,NN+NN)
#fig.add_subplot(2,1,2)
#plt.imshow(np.swapaxes(np.swapaxes(b.reshape(3,33,33),0,1),1,2)/NN*11)
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
