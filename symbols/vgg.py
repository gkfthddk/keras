from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, Concatenate
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import SimpleRNN, LSTM, GRU 
from keras.optimizers import SGD
import cv2, numpy as np

def get_symbol(input_shape,weights_path=None):
    input1 = Input(shape=input_shape)
    # group 1
    padd1_1=ZeroPadding2D((1,1))(input1)
    conv1_1=Conv2D(64, (3, 3), activation='relu')(padd1_1)
    padd1_2=ZeroPadding2D((1,1))(conv1_1)
    conv1_2=Conv2D(64, (3, 3), activation='relu')(padd1_2)
    pool1=MaxPooling2D((2,2), strides=(2,2))(conv1_2)
    # group 2
    padd2_1=ZeroPadding2D((1,1))(pool1)
    conv2_1=Conv2D(128, (3, 3), activation='relu')(padd2_1)
    padd2_2=ZeroPadding2D((1,1))(conv2_1)
    conv2_2=Conv2D(128, (3, 3), activation='relu')(padd2_2)
    padd2_3=ZeroPadding2D((1,1))(conv2_2)
    conv2_3=Conv2D(128, (3, 3), activation='relu')(padd2_3)
    padd2_4=ZeroPadding2D((1,1))(conv2_3)
    conv2_4=Conv2D(128, (3, 3), activation='relu')(padd2_4)
    pool2=MaxPooling2D((2,2), strides=(2,2))(conv2_4)
    # group 3
    padd3_1=ZeroPadding2D((1,1))(pool2)
    conv3_1=Conv2D(256, (3, 3), activation='relu')(padd3_1)
    padd3_2=ZeroPadding2D((1,1))(conv3_1)
    conv3_2=Conv2D(256, (3, 3), activation='relu')(padd3_2)
    padd3_3=ZeroPadding2D((1,1))(conv3_2)
    conv3_3=Conv2D(256, (3, 3), activation='relu')(padd3_3)
    pool3=MaxPooling2D((2,2), strides=(2,2))(conv3_3)
    padd3_4=ZeroPadding2D((1,1))(pool3)
    conv3_4=Conv2D(256, (3, 3), activation='relu')(padd3_4)
    padd3_5=ZeroPadding2D((1,1))(conv3_4)
    conv3_5=Conv2D(256, (3, 3), activation='relu')(padd3_4)
    padd3_6=ZeroPadding2D((1,1))(conv3_5)
    conv3_6=Conv2D(256, (3, 3), activation='relu')(padd3_5)
    pool3=MaxPooling2D((2,2), strides=(2,2))(conv3_6)

    flat1=Flatten()(pool3)
    flat2=Flatten()(pool1)
    concat=Concatenate(axis=1)([flat1,flat2])
    dens1=Dense(4096, activation='relu')(flat1)
    drop1=Dropout(0.5)(dens1)
    dens2=Dense(4096, activation='relu')(drop1)
    drop2=Dropout(0.5)(dens2)
    dens3=Dense(2048, activation='relu')(drop2)
    drop3=Dropout(0.5)(dens3)

    out=Dense(2,activation='softmax')(drop3)

    if weights_path:
        model.load_weights(weights_path)

    return Model(input1,out)

def rc():
    return "c"
