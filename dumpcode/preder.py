import argparse
import os

parser=argparse.ArgumentParser()
parser.add_argument("--save",type=str,default="1",help='end ratio')
parser.add_argument("--get",type=str,default="",help='end ratio')
parser.add_argument("--gpus",type=str,default="0,1,2,3",help='end ratio')
parser.add_argument("--pts",type=str,default="100,200,500,1000",help='end ratio')
parser.add_argument("--isz",type=int,default=0,help='end ratio')
parser.add_argument("--unscale",type=int,default=1,help='end ratio')
parser.add_argument("--parton",type=int,default=0,help='end ratio')
args=parser.parse_args()

saves=args.save.split(",")
pts=args.pts.split(",")
gpus=args.gpus.split(",")

if(args.get==""):
  for save in saves:
    try:
      for pt,gpu in zip(pts,gpus):
        os.system("python getpred.py --save {save} --pt {pt} --isz {isz} --gpu {gpu} --unscale {unscale} &".format(save=save.format(pt),pt=pt,gpu=gpu,isz=args.isz,unscale=args.unscale))
    except:
        print("error : python getpred.py --save {save} --pt {pt} --isz {isz} --gpu {gpu} --unscale {unscale}".format(save=save.format(pt),pt=pt,gpu=gpu,isz=args.isz,unscale=args.unscale))
else:  
  for save in saves:
    try:
      for pt,gpu in zip(pts,gpus):
        os.system("python getauc.py --save {save} --pt {pt} --get {get} --parton {parton} &".format(save=save.format(pt),pt=pt,get=args.get,parton=args.parton))
    except:
        print("error : python getauc.py --save {save} --pt {pt} --get {get} --parton {parton} &".format(save=save.format(pt),pt=pt,get=args.get,parton=args.parton))
os.system("echo 0")
