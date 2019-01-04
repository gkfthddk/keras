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
parser.add_argument("--name",default="",help='name including')

args=parser.parse_args()

start=datetime.datetime.now()
files=os.listdir("save")
files.sort()
for f in files:
  if args.name in f:
    pass
  else:
    continue
  savename="save/"+f
  dijetacc=0
  zjetacc=0
  dijetepoch=0
  try:
    history=open(savename+"/history")
    indx=eval(history.readline())
    acc=eval(history.readline())['val_acc'][indx]
    print "{}\t{:.3f}".format(f,acc)
  except:
    pass

print datetime.datetime.now()-start
