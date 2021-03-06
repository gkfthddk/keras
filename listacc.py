import random
import datetime
import argparse
import sys
import os
import subprocess
# python run.py --network vgg --end 0.1 --batch-size 100 --num-epochs 10 --gpus "1" --optimizer sgd
#python run.py --end 1 --batch-size 100 --num-epochs 20 --gpus "1" --network vgg --optimizer sgd --rat 0.49
#python run.py --end 1 --batch-size 100 --num-epochs 30 --gpus "0" --network vgg --optimizer sgd --train w --rat 0.8

parser=argparse.ArgumentParser()
parser.add_argument("--inc",default="",help='name including')
parser.add_argument("--exc",default="",help='name excluding')

args=parser.parse_args()

start=datetime.datetime.now()
files=os.listdir("save")
files.sort()
include=args.inc.split(",")
exclude=args.exc.split(",")

for f in files:
  check=0
  if(exclude!=[""]):
    for na in exclude:
      if na in f:
        check=1
        break
  if check==1:
    continue
  for na in include:
    if na in f:
      pass
    else:
      check=1
      break
  if check==1:
    continue
  savename="save/"+f
  acc=[]
  try:
    history=open(savename+"/history")
    indx=[eval(i) for i in history.readline().split(' ')]
    hist=eval(history.readline())
    acc.append(max(hist['val1_acc']))
    try:
      acc.append(max(hist['val2_acc']))
      acc.append(max(hist['val3_acc']))
    except:pass
    line="{}".format(f)
    for i,j in zip(indx,acc):
      line+="\t{:.3f}\t{}".format(j,i)
    print(line)
  except:
    pass

print datetime.datetime.now()-start
