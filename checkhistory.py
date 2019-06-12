'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function
import argparse
import numpy as np
import os
from importlib import import_module
import matplotlib.pyplot as plt
plt.switch_backend('agg')

batch_size = 100000

parser=argparse.ArgumentParser()
parser.add_argument("--end",type=float,default=100000.,help='end ratio')
parser.add_argument("--epoch",type=int,default=None,help='epoch')
parser.add_argument("--save",type=str,default="ten100grucnn",help='rch')
parser.add_argument("--rc",type=str,default='rc',help='rnn or cnn')
parser.add_argument("--gpu",type=int,default=0,help='gpu')
parser.add_argument("--pt",type=int,default=100,help='pt range pt~pt*1.1')
parser.add_argument("--ptmin",type=float,default=0.,help='pt range pt~pt*1.1')
parser.add_argument("--ptmax",type=float,default=2.,help='pt range pt~pt*1.1')
parser.add_argument("--isz",type=int,default=0,help='0 or z or not')
parser.add_argument("--channel",type=int,default=64,help='sequence channel')
parser.add_argument("--order",type=int,default=1,help='pt ordering')
parser.add_argument("--eta",type=float,default=0.,help='end ratio')
parser.add_argument("--etabin",type=float,default=2.4,help='end ratio')

args=parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
# input image dimensions
img_rows, img_cols = 33, 33

# the data, shuffled and split between train and test sets
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

#if K.image_data_format() == 'channels_first':
#    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#    input_shape = (1, img_rows, img_cols)
#else:
#x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape1= (9,33,33)
input_shape2= (20,10)

savename="save/"+str(args.save)
history=open(savename+"/history").readlines()
try:
  hist=eval(history[0])
  a=hist['val1_auc']
except:
  hist=eval(history[1])
vzjdata="Data/zj_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
vjjdata="Data/jj_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
vzqdata="Data/zq_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
vzgdata="Data/zg_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
vqqdata="Data/qq_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
vggdata="Data/gg_pt_{0}_{1}.root".format(args.pt,int(args.pt*1.1))
from sklearn.metrics import roc_auc_score, auc, roc_curve
if(args.isz==0):iii=1
if(args.isz==1):iii=2
if(args.isz==-1):iii=3
print(savename,len(hist['val{}_auc'.format(iii)]))
#epoch=hist['val{}_auc'.format(iii)].index(max(hist['val{}_auc'.format(iii)]))+1
