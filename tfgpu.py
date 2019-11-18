#!/usr/bin/python2.7
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
import os
gpu=6
if(gpu!=-1):
  os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
  config =tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction=0.3

  set_session(tf.Session(config=config))
with tf.Session() as sess:
  devices = sess.list_devices()
  print(devices)


print("ernd")
