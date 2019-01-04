'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import keras
from keras import backend as K
from iter import *
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from importlib import import_module
import matplotlib.pyplot as plt
plt.switch_backend('agg')
config =tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.2
set_session(tf.Session(config=config))

batch_size = 256

parser=argparse.ArgumentParser()
parser.add_argument("--rat",type=float,default=0.6,help='ratio for weak qg batch')
parser.add_argument("--end",type=float,default=1.,help='end ratio')
parser.add_argument("--epoch",type=int,default=10,help='epoch')
parser.add_argument("--save",type=str,default="qg100lstmcnn",help='rch')
parser.add_argument("--rc",type=str,default='rc',help='rnn or cnn')
args=parser.parse_args()

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
input_shape1= (3,33,33)
input_shape2= (20,4)

#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.Adadelta(),
#              metrics=['accuracy'])
#model=keras.models.load_model('save/fullydijetsame_10')
savename="save/"+str(args.save)
epoch=eval(open(savename+"/history").readline())+1
model=keras.models.load_model(savename+"/check_"+str(epoch))

if(len(model._feed_inputs)==2):rc="rc"
elif(len(model._feed_inputs[0]._keras_shape)==3):rc="r"
elif(len(model._feed_inputs[0]._keras_shape)==4):rc="c"

test=wkiter(["/scratch/yjdata/quark100_img.root","/scratch/yjdata/gluon100_img.root"],batch_size=batch_size,begin=3./5.,end=args.end*1./5.+3./5.,rc=rc)
gen=test.next()
from sklearn.metrics import roc_auc_score, auc,precision_recall_curve,roc_curve,average_precision_score
x=[]
y=[]
g=[]
q=[]
entries=600
batch_num=batch_size
print ("test",test.totalnum())
for j in range(entries):
	a,c=next(gen)
	b=model.predict(a,verbose=0)[:,0]
	x=np.append(x,np.array(c[:,0]))
	y=np.append(y,b)
	for i in range(batch_num):
		if(c[i][0]==1):
			g.append(b[i])
		else:
			q.append(b[i])
plt.figure(1)
plt.hist(q,bins=50,weights=np.ones_like(q),histtype='step',alpha=0.7,label='quark')
plt.hist(g,bins=50,weights=np.ones_like(g),histtype='step',alpha=0.7,label='gluon')
plt.legend(loc="upper center")
plt.savefig(savename+"/out.png")
f=open(savename+"/out.dat",'w')
f.write(str(q)+"\n")
f.write(str(g))
f.close()
t_fpr,t_tpr,_=roc_curve(x,y)
t_fnr=1-t_fpr
test_auc=np.around(auc(t_fpr,t_tpr),4)
plt.figure(2)
plt.plot(t_tpr,t_fnr,alpha=0.5,label="AUC={}".format(test_auc),lw=2)
plt.legend(loc='lower left')
os.system("rm "+savename+"/roc*.png")
plt.savefig(savename+"/roc"+str(test_auc)+".png")
f=open(savename+"/roc.dat",'w')
f.write(str(t_tpr.tolist())+"\n")
f.write(str(t_fnr.tolist()))
f.close()
#print(b,c)

