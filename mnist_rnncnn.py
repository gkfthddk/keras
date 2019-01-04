'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
from __future__ import print_function
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import SimpleRNN, LSTM
from keras.layers.wrappers import TimeDistributed
from keras import initializers
from keras import backend as K
from keras.utils import plot_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from importlib import import_module
config =tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.8
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))


batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

rx_train = x_train.reshape(x_train.shape[0], -1, 1)
rx_test = x_test.reshape(x_test.shape[0], -1, 1)
rx_train = rx_train.astype('float32')
rx_test = rx_test.astype('float32')
rx_train /= 255
rx_test /= 255

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
rx_train = x_train.reshape(x_train.shape[0], -1, 1)
rx_test = x_test.reshape(x_test.shape[0], -1, 1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
input_shape=Input(shape=input_shape)
#rinput_shape=Reshape(rx_train.shape[1:])(input_shape)
rinput_shape=Input(rx_train.shape[1:])
conv_1_1=Conv2D(32,kernel_size=(1,3),activation='relu')(input_shape)
#conv_1_1=Conv2D(32,kernel_size=(1,3),activation='relu')(Input(shape=input_shape))
conv_1_2=Conv2D(32,kernel_size=(3,1),activation='relu')(conv_1_1)
conv_2_1=Conv2D(64,kernel_size=(3,1),activation='relu')(conv_1_2)
conv_2_2=Conv2D(64,kernel_size=(3,1),activation='relu')(conv_2_1)
pool_1=MaxPooling2D(pool_size=(2,2))(conv_2_2)
drop=Dropout(0.25)(pool_1)
flatten=Flatten()(drop)
rnn=SimpleRNN(20,kernel_initializer=initializers.RandomNormal(stddev=0.001),
              recurrent_initializer=initializers.Identity(gain=1.0),
              activation='relu')(rinput_shape)
dense=Dense(128,activation='relu')(flatten)
rdense=Dense(128,activation='relu')(rnn)
#concat=Dense(128,activation='relu')(flatten)
concat=keras.layers.concatenate([dense,rdense],axis=1)
dense2=Dense(128,activation='relu')(Dropout(0.5)(concat))
out=Dense(num_classes,activation='softmax')(Dropout(0.5)(dense2))

model=Model([input_shape,rinput_shape],out)
plot_model(model,to_file='rnncnn.png')
print("####plot#####")
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
