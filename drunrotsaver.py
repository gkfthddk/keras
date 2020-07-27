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
  infile=rt.TFile("/home/yulee/geant4/tester/analysis/{}{}sum.root".format(name,pt),'read')
  event=infile.Get("event")
  img_e_s=[]
  img_e_c=[]
  img_n_s=[]
  img_n_c=[]
  image=[]
  voxel=[]
  """xbin=15
  xmin=-14
  xmax=16
  ybin=15
  ymin=-7
  ymax=8"""
  #xbin=17
  #ybin=17
  #zbin=17
  xbin=22
  #xbin=30 # rot8ug
  ybin=23
  zbin=23
  xmin=0
  #xmax=1150
  #xmax=1525
  #xmax=1000 # rot8ug
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
    #ymin=-0.0058
    #ymax=0.006
    #zmin=0.006
    #zmax=0.018
    #ymin=-26
    #ymax=26
    #zmin=24
    #zmax=76
  xsize=1.*(xmax-xmin)/xbin
  ysize=1.*(ymax-ymin)/ybin
  zsize=1.*(zmax-zmin)/zbin
  ycent=ymax/2.+ymin/2.
  zcent=zmax/2.+zmin/2.
  for num_entry in range(event.GetEntries()):
    event.GetEntry(num_entry)
    for rot in range(1):
      rotation_angle = 0.
      cosval = np.cos(rotation_angle)
      sinval = np.sin(rotation_angle)
      if("j" in name):
        oneimgyzx_e2_s=np.zeros((zbin,ybin,xbin+1),dtype='float32')
        oneimgyzx_e2_c=np.zeros((zbin,ybin,xbin+1),dtype='float32')
        oneimgyz_e2_s=np.zeros((zbin,ybin),dtype='float32')
        oneimgyz_e2_c=np.zeros((zbin,ybin),dtype='float32')
        oneimgyz_n2_s=np.zeros((zbin,ybin),dtype='float32')
        oneimgyz_n2_c=np.zeros((zbin,ybin),dtype='float32')
        oneimgyx_e2_s=np.zeros((xbin+1,ybin),dtype='float32')
        oneimgyx_n2_s=np.zeros((xbin+1,ybin),dtype='float32')
        oneimgzx_e2_s=np.zeros((xbin+1,zbin),dtype='float32')
        oneimgzx_n2_s=np.zeros((xbin+1,zbin),dtype='float32')
      oneimgyzx_e_s=np.zeros((zbin,ybin,xbin+1),dtype='float32')
      oneimgyzx_e_c=np.zeros((zbin,ybin,xbin+1),dtype='float32')
      oneimgyz_e_s=np.zeros((zbin,ybin),dtype='float32')
      oneimgyz_e_c=np.zeros((zbin,ybin),dtype='float32')
      oneimgyz_n_s=np.zeros((zbin,ybin),dtype='float32')
      oneimgyz_n_c=np.zeros((zbin,ybin),dtype='float32')
      oneimgyx_e_s=np.zeros((xbin+1,ybin),dtype='float32')
      oneimgyx_n_s=np.zeros((xbin+1,ybin),dtype='float32')
      oneimgzx_e_s=np.zeros((xbin+1,zbin),dtype='float32')
      oneimgzx_n_s=np.zeros((xbin+1,zbin),dtype='float32')
      for i in range(len(event.fiber_iscerenkov)):
        xpre=event.fiber_depth[i]
        ypre=event.fiber_phi[i]-ycent
        zpre=event.fiber_eta[i]-zcent
        jet2=0
        if("j" in name and abs(ypre)>rt.TMath.Pi()/2):
          jet2=1
          if(ypre<0):
            ypre=rt.TMath.Pi()+ypre
          else:
            ypre=ypre-rt.TMath.Pi()
        x=xpre
        y=cosval*ypre-sinval*zpre+ycent
        z=sinval*ypre+cosval*zpre+zcent
        xindex=-1
        yindex=-1
        zindex=-1
        if(xmax>x and xmin<=x):
          xindex=int((x-xmin)/xsize)
        if(ymax>y and ymin<=y):
          yindex=int((y-ymin)/ysize)
        if(zmax>z and zmin<=z):
          zindex=int((z-zmin)/zsize)
        if(yindex!=-1 and zindex!=-1):
          if(bool(event.fiber_iscerenkov[i])==False):
            if(jet2==0):
              oneimgyz_e_s[zindex,yindex]+=event.fiber_ecor[i]
              oneimgyz_n_s[zindex,yindex]+=event.fiber_n[i]
              if(xindex!=-1):
                oneimgyx_e_s[xindex,yindex]+=event.fiber_ecor[i]
                oneimgyx_n_s[xindex,yindex]+=event.fiber_n[i]
                oneimgzx_e_s[xindex,zindex]+=event.fiber_ecor[i]
                oneimgzx_n_s[xindex,zindex]+=event.fiber_n[i]
              if(x==0):
                xindex=1
              else:
                xindex+=1
              if(xindex!=-1):
                oneimgyzx_e_s[zindex,yindex,xindex]+=event.fiber_ecor[i]
            else:
              oneimgyz_e2_s[zindex,yindex]+=event.fiber_ecor[i]
              oneimgyz_n2_s[zindex,yindex]+=event.fiber_n[i]
              if(xindex!=-1):
                oneimgyx_e2_s[xindex,yindex]+=event.fiber_ecor[i]
                oneimgyx_n2_s[xindex,yindex]+=event.fiber_n[i]
                oneimgzx_e2_s[xindex,zindex]+=event.fiber_ecor[i]
                oneimgzx_n2_s[xindex,zindex]+=event.fiber_n[i]
              if(x==0):
                xindex=1
              else:
                xindex+=1
              if(xindex!=-1):
                oneimgyzx_e2_s[zindex,yindex,xindex]+=event.fiber_ecor[i]
          else:
            if(jet2==0):
              oneimgyz_e_c[zindex,yindex]+=event.fiber_ecor[i]
              oneimgyz_n_c[zindex,yindex]+=event.fiber_n[i]
              if(x==0):
                xindex=1
              else:
                xindex+=1
              if(xindex!=-1):
                oneimgyzx_e_c[zindex,yindex,xindex]+=event.fiber_ecor[i]
            else:
              oneimgyz_e2_c[zindex,yindex]+=event.fiber_ecor[i]
              oneimgyz_n2_c[zindex,yindex]+=event.fiber_n[i]
              if(x==0):
                xindex=1
              else:
                xindex+=1
              if(xindex!=-1):
                oneimgyzx_e2_c[zindex,yindex,xindex]+=event.fiber_ecor[i]
  
      image.append([np.array(oneimgyz_e_s,dtype='float32'),np.array(oneimgyz_e_c,dtype='float32'),np.array(oneimgyz_n_s,dtype='float32'),np.array(oneimgyz_n_c,dtype='float32'),np.array(oneimgyx_e_s,dtype='float32'),np.array(oneimgzx_e_s,dtype='float32'),np.array(oneimgyx_n_s,dtype='float32'),np.array(oneimgzx_n_s,dtype='float32')])
      voxel.append(np.array([oneimgyzx_e_s,oneimgyzx_e_c],dtype='float32'))
      if("j" in name):
        image.append([np.array(oneimgyz_e2_s,dtype='float32'),np.array(oneimgyz_e2_c,dtype='float32'),np.array(oneimgyz_n2_s,dtype='float32'),np.array(oneimgyz_n2_c,dtype='float32'),np.array(oneimgyx_e2_s,dtype='float32'),np.array(oneimgzx_e2_s,dtype='float32'),np.array(oneimgyx_n2_s,dtype='float32'),np.array(oneimgzx_n2_s,dtype='float32')])
        voxel.append(np.array([oneimgyzx_e2_s,oneimgyzx_e2_c],dtype='float32'))
  
    #img_e_s.append(event.img_e_s)
    #img_e_c.append(event.img_e_c)
    #img_n_s.append(event.img_n_s)
    #img_n_c.append(event.img_n_c)
  infile.Close()
  imgset[name]=np.array(image,dtype='float32')
  voxels[name]=np.array(voxel,dtype='float32')
#np.savez_compressed("/home/yulee/keras/rot23ug{}img".format(pt),uj=imgset["uj"],gj=imgset["gj"])
#np.savez_compressed("/home/yulee/keras/rot23ug{}vox".format(pt),uj=voxels["uj"],gj=voxels["gj"])
np.savez_compressed("/home/yulee/keras/egp{}img".format(pt),el=imgset["el"],ga=imgset["ga"],pi=imgset["pi"])
np.savez_compressed("/home/yulee/keras/egp{}vox".format(pt),el=voxels["el"],ga=voxels["ga"],pi=voxels["pi"])
print(datetime.now()-now)
