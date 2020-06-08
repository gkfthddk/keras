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
  namelist=["uj","gj","pi","el","ga"]
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
  xbin=23
  ybin=23
  zbin=23
  xmin=0
  #xmax=1150
  xmax=1525
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
  xsize=1.*(xmax-xmin)/ybin
  ysize=1.*(ymax-ymin)/ybin
  zsize=1.*(zmax-zmin)/zbin
  phist=np.zeros((zbin,ybin))
  for num_entry in range(event.GetEntries()):
    event.GetEntry(num_entry)
    if("j" in name):
      oneimgyzx_e2_s=np.zeros((zbin,ybin,xbin))
      oneimgyzx_e2_c=np.zeros((zbin,ybin,xbin))
      oneimgyz_e2_s=np.zeros((zbin,ybin))
      oneimgyz_e2_c=np.zeros((zbin,ybin))
      oneimgyz_n2_s=np.zeros((zbin,ybin))
      oneimgyz_n2_c=np.zeros((zbin,ybin))
      oneimgyx_e2_s=np.zeros((xbin,ybin))
      oneimgyx_n2_s=np.zeros((xbin,ybin))
      oneimgzx_e2_s=np.zeros((xbin,zbin))
      oneimgzx_n2_s=np.zeros((xbin,zbin))
    oneimgyzx_e_s=np.zeros((zbin,ybin,xbin))
    oneimgyzx_e_c=np.zeros((zbin,ybin,xbin))
    oneimgyz_e_s=np.zeros((zbin,ybin))
    oneimgyz_e_c=np.zeros((zbin,ybin))
    oneimgyz_n_s=np.zeros((zbin,ybin))
    oneimgyz_n_c=np.zeros((zbin,ybin))
    oneimgyx_e_s=np.zeros((xbin,ybin))
    oneimgyx_n_s=np.zeros((xbin,ybin))
    oneimgzx_e_s=np.zeros((xbin,zbin))
    oneimgzx_n_s=np.zeros((xbin,zbin))
    for i in range(len(event.fiber_iscerenkov)):
      x=event.fiber_depth[i]
      y=event.fiber_phi[i]
      z=event.fiber_eta[i]
      jet2=0
      if("j" in name and abs(y)>rt.TMath.Pi()/2):
        jet2=1
        if(y<0):
          y=rt.TMath.Pi()+y
        else:
          y=y-rt.TMath.Pi()
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
              oneimgyzx_e_s[zindex,yindex,xindex]+=event.fiber_ecor[i]
          else:
            oneimgyz_e2_s[zindex,yindex]+=event.fiber_ecor[i]
            oneimgyz_n2_s[zindex,yindex]+=event.fiber_n[i]
            if(xindex!=-1):
              oneimgyx_e2_s[xindex,yindex]+=event.fiber_ecor[i]
              oneimgyx_n2_s[xindex,yindex]+=event.fiber_n[i]
              oneimgzx_e2_s[xindex,zindex]+=event.fiber_ecor[i]
              oneimgzx_n2_s[xindex,zindex]+=event.fiber_n[i]
              oneimgyzx_e2_s[zindex,yindex,xindex]+=event.fiber_ecor[i]
        else:
          if(jet2==0):
            oneimgyz_e_c[zindex,yindex]+=event.fiber_ecor[i]
            oneimgyz_n_c[zindex,yindex]+=event.fiber_n[i]
            if(xindex!=-1):
              oneimgyzx_e_c[zindex,yindex,xindex]+=event.fiber_ecor[i]
          else:
            oneimgyz_e2_c[zindex,yindex]+=event.fiber_ecor[i]
            oneimgyz_n2_c[zindex,yindex]+=event.fiber_n[i]
            if(xindex!=-1):
              oneimgyzx_e2_c[zindex,yindex,xindex]+=event.fiber_ecor[i]
  
    image.append([np.array(oneimgyz_e_s),np.array(oneimgyz_e_c),np.array(oneimgyz_n_s),np.array(oneimgyz_n_c),np.array(oneimgyx_e_s),np.array(oneimgzx_e_s),np.array(oneimgyx_n_s),np.array(oneimgzx_n_s)])
    voxel.append(np.array([oneimgyzx_e_s,oneimgyzx_e_c]))
    if("j" in name):
      image.append([np.array(oneimgyz_e2_s),np.array(oneimgyz_e2_c),np.array(oneimgyz_n2_s),np.array(oneimgyz_n2_c),np.array(oneimgyx_e2_s),np.array(oneimgzx_e2_s),np.array(oneimgyx_n2_s),np.array(oneimgzx_n2_s)])
      voxel.append(np.array([oneimgyzx_e2_s,oneimgyzx_e2_c]))
  
    #img_e_s.append(event.img_e_s)
    #img_e_c.append(event.img_e_c)
    #img_n_s.append(event.img_n_s)
    #img_n_c.append(event.img_n_c)
  infile.Close()
  imgset[name]=np.array(image)
  voxels[name]=np.array(voxel)
np.savez_compressed("/home/yulee/keras/side023{}img".format(pt),**imgset)
np.savez_compressed("/home/yulee/keras/side023{}vox".format(pt),**voxels)
