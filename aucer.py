import os
import sys
import subprocess
import ROOT as rt
import numpy as np
import scipy.stats
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--save",type=str,default="1",help='io')
parser.add_argument("--name",type=str,default="1",help='io')
args=parser.parse_args()
ptmin=0
ptmax=0
saves=args.save.split(",")
names=args.name.split(",")
frac={}
#saves=['asuqqcnn{}ptonlyptparton','asuqqcnn{}ptonlyqgpt','asuqqrnn{}ptonly21ptparton','asuqqrnn{}ptonly21qgpt']
#names=["cnndijet","cnnqg","rnndijet","rnnqg"]
#frac={"cnndijet":[],"cnnqg":[],"rnndijet":[],"rnnqg":[]}
for name in names:
  frac[name]=[]
#frac2={"cnndijet":[],"cnnzjet":[],"cnnqg":[],"rnndijet":[],"rnnzjet":[],"rnnqg":[]}
for pt in [100,200,500,1000]:
  for save,name in zip(saves,names):
    dic=eval(open("aucs/"+save.format(pt)).readline())
    if("qg" in name or "dijet" in name):
      frac[name].append(dic["dijet"])
      #frac2[name].append(dic["dijetsem"])
    if("zjet" in name):
      frac[name].append(dic["Z+jet"])
      #frac2[name].append(dic["Z+jetsem"])
for k in frac:
  a=" {} |"*5
  a="|"+a
  print(a.format(k,*frac[k]))
#for k in frac2:
#  a=" {} |"*5
#  a="|"+a
#  print(a.format(k,*frac2[k]))
