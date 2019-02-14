from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, Concatenate
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, LocallyConnected2D
from keras.layers import SimpleRNN, LSTM, GRU 
from keras.optimizers import SGD
import cv2, numpy as np

def get_symbol(input_shape,weights_path=None):
    input1 = Input(shape=input_shape)
    # group 1
    padd1_1=ZeroPadding2D((1,1))(input1)
    conv1_1=LocallyConnected2D(16, (3, 3), activation='relu')(padd1_1)
    pool1=MaxPooling2D((2,2), strides=(2,2))(conv1_1)
    # group 2
    conv2_1=LocallyConnected2D(32, (3, 3), activation='relu')(pool1)
    pool2=MaxPooling2D((2,2), strides=(2,2))(conv2_1)

    flat1=Flatten()(pool2)
    dens1=Dense(2048, activation='relu')(flat1)
    drop1=Dropout(0.5)(dens1)
    out=Dense(2,activation='softmax')(drop1)

    if weights_path:
        model.load_weights(weights_path)

    return Model(input1,out)

def rc():
    return "c"
