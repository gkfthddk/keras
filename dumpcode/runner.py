import argparse
import os

parser=argparse.ArgumentParser()
parser.add_argument("--save",type=str,default="1",help='end ratio')
parser.add_argument("--get",type=str,default="ptcut",help='end ratio')
parser.add_argument("--gpu",type=str,default="0,1,2,3",help='end ratio')
parser.add_argument("--pt",type=str,default="100,200,500,1000",help='end ratio')
parser.add_argument("--isz",type=int,default=0,help='end ratio')
parser.add_argument("--etabin",type=float,default=2.4,help='end ratio')
parser.add_argument("--unscale",type=int,default=1,help='end ratio')
parser.add_argument("--parton",type=int,default=0,help='end ratio')
parser.add_argument("--run",type=int,default=0,help='end ratio')

args=parser.parse_args()

save=args.save
pt=eval(args.pt)
gpu=args.gpu

if(pt==100):
  ptmin=0.815
  ptmax=1.159
if(pt==200):
  ptmin=0.819
  ptmax=1.123
if(pt==500):
  ptmin=0.821
  ptmax=1.093
if(pt==1000):
  ptmin=0.8235
  ptmax=1.076
if(args.run<1):
  os.system("python partonrun.py --pt {pt} --save {save} --end 100000 --epochs 50 --gpu {gpu} --ptmin {ptmin} --ptmax {ptmax} --batch_size 1000000 --isz {isz} --etabin {etabin}".format(save=save.format(pt),pt=pt,gpu=gpu,isz=args.isz,etabin=args.etabin,ptmin=ptmin,ptmax=ptmax))
if(args.run<2):
  os.system("python partonpred.py --save {save} --pt {pt} --isz {isz} --gpu {gpu} ".format(save=save.format(pt),pt=pt,gpu=gpu,isz=args.isz))
if(args.run<3):
  os.system("python getauc.py --save {save} --pt {pt} --get {get} --parton {parton} ".format(save=save.format(pt),pt=pt,get=args.get,parton=args.parton))
