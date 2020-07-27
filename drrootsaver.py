#!/usr/bin/python2.7
import os
import sys
import subprocess
import numpy as np
#from jetiter import * #image original
from datetime import datetime
import random
import argparse
import random
import warnings
import ROOT as rt
import math
from array import array
from sklearn import preprocessing
parser=argparse.ArgumentParser()
parser.add_argument("--etabin",type=float,default=1.,help='end ratio')
parser.add_argument("--pt",type=int,default=200,help='end ratio')
parser.add_argument("--stride",type=int,default=2,help='end ratio')
args=parser.parse_args()
ptmin=0
ptmax=0
now=datetime.now()
pt=args.pt
imgset={}
voxels={}
if(pt==20):
  namelist=["pi","el","ga"]
if(pt==50):
  namelist=["pi","el","ga"]
for name in namelist:#,'uj','gj']:
  infile=rt.TFile("/home/yulee/geant4/tester/analysis/{}{}img.root".format(name,pt),'read')
  event=infile.Get("event")
  image=[]
  voxel=[]
  xbin=22#22+1
  ybin=23
  zbin=23
  xmin=0
  xmax=1496
  count=0
  if("j" in name):
    ymin=-0.0221*ybin/2*2 # 2 tower in one bin #*4 for bigger bin size
    ymax=0.0221*ybin/2*2
    zmin=-0.0221*zbin/2*2
    zmax=0.0221*zbin/2*2
    #ymin=-0.5
    #ymax=0.5
    #zmin=-0.5
    #zmax=0.5
  else:
    ymin=-0.008
    ymax=0.0081
    zmin=0.003
    zmax=0.0191
  for num_entry in range(event.GetEntries()):
    event.GetEntry(num_entry)
    image.append([np.array(event.image_ecor_s,dtype='float32').reshape((23,23)),np.array(event.image_ecor_c,dtype='float32').reshape((23,23)),np.array(event.image_n_s,dtype='float32').reshape((23,23)),np.array(event.image_n_c,dtype='float32').reshape((23,23))])
  
  infile.Close()
  imgset[name]=np.array(image,dtype='float32')
#np.savez_compressed("/home/yulee/keras/rot23ug{}img".format(pt),uj=imgset["uj"],gj=imgset["gj"])
#np.savez_compressed("/home/yulee/keras/rot23ug{}vox".format(pt),uj=voxels["uj"],gj=voxels["gj"])
np.savez_compressed("/home/yulee/keras/egp{}img".format(pt),el=imgset["el"],ga=imgset["ga"],pi=imgset["pi"])
print(datetime.now()-now)
