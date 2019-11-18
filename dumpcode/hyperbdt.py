import xgboost as xgb
from sklearn.metrics import roc_auc_score, auc, roc_curve
import pandas as pd
import numpy as np
import pickle
from xiter import *
import argparse
import matplotlib.pyplot as plt

data=pd.read_csv('xgb/random-search.csv')

parser=argparse.ArgumentParser()
parser.add_argument("--end",type=float,default=100000.,help='end ratio')
parser.add_argument("--save",type=str,default="random-search.csv",help='save name')
parser.add_argument("--right",type=str,default="/scratch/yjdata/gluon100_img",help='which train sample (qq,gg,zq,zg)')
parser.add_argument("--pt",type=int,default=200,help='pt range pt~pt*1.1')
parser.add_argument("--ptmin",type=float,default=0.,help='pt range pt~pt*1.1')
parser.add_argument("--ptmax",type=float,default=2.,help='pt range pt~pt*1.1')
parser.add_argument("--batch_size",type=int,default=100000,help='batch_size')
parser.add_argument("--gpu",type=int,default=0,help='gpu number')
parser.add_argument("--isz",type=int,default=0,help='0 or z or not')
parser.add_argument("--channel",type=int,default=30,help='sequence channel')
parser.add_argument("--order",type=int,default=1,help='pt ordering')
parser.add_argument("--eta",type=float,default=0.,help='end ratio')
parser.add_argument("--etabin",type=float,default=1.,help='end ratio')

args=parser.parse_args()
rc=""
onehot=0
csv=pd.read_csv("xgb/{}-{}.csv".format(args.save,args.pt))
#csv=pd.read_csv(args.save)
best=csv.loc[csv["mean_test_score"].idxmax()]

print(best)
