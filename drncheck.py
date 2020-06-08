import numpy as np
f=open("drc","r")
drc=f.readlines()
f.close()
ch={}
for i in drc:
  
  ds=i.replace("\n","").split(":")
  #print(ds[0])
  if(not ds[0] in ch):
    ch[ds[0]]=[]
  ch[ds[0]].append(float(ds[1]))

for j in ch.keys():
  print(j,np.mean(ch[j]))
