'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1" 
import argparse
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Embedding
from keras.layers import Conv2D, MaxPooling2D, SimpleRNN
from keras import backend as K
from keras.utils import plot_model
from iter import *
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from importlib import import_module
import datetime
start=datetime.datetime.now()
config =tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.4
set_session(tf.Session(config=config))

def weakloss(ytrue,ypred):
	a=K.sum(ypred)/ypred.shape[0] 
	b=K.sum(ytrue)/ypred.shape[0]
	loss=a - b
	print(type(loss))
	loss=K.square(loss)
	return loss
def mean_squared_error(y_true,y_pred):
	return K.mean(K.square(y_pred - y_true),axis=-1)
batch_size = 128
num_classes = 2 
epochs = 10 

parser=argparse.ArgumentParser()
parser.add_argument("--rat",type=float,default=0.6,help='ratio for weak qg batch')
parser.add_argument("--end",type=float,default=1.,help='end ratio')
parser.add_argument("--save",type=str,default="test_",help='save name')
parser.add_argument("--network",type=str,default="asym",help='network name on symbols/')
parser.add_argument("--left",type=str,default="qq",help='which train sample (qq,gg,zq,zg)')
parser.add_argument("--right",type=str,default="gg",help='which train sample (qq,gg,zq,zg)')
parser.add_argument("--ztest",type=int,default=0,help='true get zjet test')
parser.add_argument("--loss",type=str,default="categorical_crossentropy",help='network name on symbols/')
args=parser.parse_args()

# input image dimensions
img_rows, img_cols = 33, 33

input_shape = (3,33,33)

if(args.loss=="weakloss"):args.loss=weakloss
net=import_module('symbols.'+args.network)
model=net.get_symbol(input_shape,num_classes)
plot_model(model,to_file='ctest.png')
print("### plot done ###")
#model.compile(loss='mean_squared_error',
model.compile(loss=args.loss,
              optimizer=keras.optimizers.SGD(),
	      metrics=['accuracy'])
"""model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])
"""
#ab=int(len(x_train)/10)

train=wkiter(["/scratch/yjdata/quark100_img.root","/scratch/yjdata/gluon100_img.root"],batch_size=batch_size,end=args.end,istrain=1)
#train=wkiter(["root/new/q"+str(int(args.rat*100))+"img.root","root/new/g"+str(int(args.rat*100))+"img.root"],batch_size=batch_size,end=5./7.,istrain=1,friend=0)

#savename='save/'+str(args.save)+str(args.rat)
savename='save'
os.system("mkdir "+savename)
print ("train",train.totalnum(),"eval")
#logger=keras.callbacks.CSVLogger(savename+'/log.log',append=True)
#logger=keras.callbacks.TensorBoard(log_dir=savename+'/logs',histogram_freq=0, write_graph=True , write_images=True, batch_size=20)
history=0
#checkpoint=keras.callbacks.ModelCheckpoint(filepath=savename+'/_{epoch}',monitor='val_loss',verbose=0,save_best_only=False,mode='auto',period=1)

#history=model.fit_generator(train.next(),steps_per_epoch=train.totalnum(),epochs=epochs,verbose=1,callbacks=[])
model.fit_generator(train.next(),steps_per_epoch=train.totalnum(),epochs=epochs,verbose=1)

#gen=train.next()
#a,b=next(gen)
#history=model.fit(a,b,epochs=epochs,verbose=1,callbacks=[logger])

#train.reset()
#test.reset()

"""print(history)
f=open(savename+'/history','w')
try:
  one=history.history['val_acc'].index(max(history.history['val_acc']))
  f.write(str(one)+'\n')
  print(one)
  for i in range(epochs):
    if(i!=one):os.system("rm "+savename+"/_"+str(i+1))
except:pass
f.write(str(history.history))
f.close()"""
print (datetime.datetime.now()-start)
