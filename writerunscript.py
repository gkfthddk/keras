#!/usr/bin/python2.7
import os
os.system("cp run.co duels/run.co")
f=open("run.co","w")
base="""
#rsect and dsect in symbols for ernntest 0 to 3
Universe = vanilla
getenv = True
Executable = gendual.py 
initialdir = /home/yulee/keras
should_transfer_files=yes
transfer_input_files=symbols
request_gpus = 1
"""
f.write(base)
for i in range(4):
  for j in range(4):
    que="""output = condor/goop{r}{d}.out
error = condor/goop{r}{d}.error
Log = condor/goop{r}{d}.log
Arguments = --save gentest{r}{d}500 --network ernntest --pt 500 --epoch 10 --stride 2 --gpu 2 --pred -1 --seed non --opt adam --rsect {r} --dsect {d} --memo 'rsect {r} dsect {d}'
Queue
""".format(r=i,d=j)
    f.write(que)
f.close()
