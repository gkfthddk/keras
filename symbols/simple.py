from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, Concatenate
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import SimpleRNN, LSTM, GRU 
from keras.optimizers import SGD
import cv2, numpy as np

def get_symbol(input_shape1,input_shape2,weights_path=None):
    input1 = Input(shape=input_shape1)
    input2 = Input(shape=input_shape2)
    # group 1
    padd1_1=ZeroPadding2D((1,0))(input1)
    conv1_1=Conv2D(64, (3, 1), activation='relu')(padd1_1)
    padd1_2=ZeroPadding2D((0,1))(conv1_1)
    conv1_2=Conv2D(64, (1, 3), activation='relu')(padd1_2)
    pool1=MaxPooling2D((2,2), strides=(2,2))(conv1_2)
    # group 2
    padd2_1=ZeroPadding2D((1,0))(pool1)
    conv2_1=Conv2D(128, (3, 1), activation='relu')(padd2_1)
    padd2_2=ZeroPadding2D((0,1))(conv2_1)
    conv2_2=Conv2D(128, (1, 3), activation='relu')(padd2_2)
    padd2_3=ZeroPadding2D((1,0))(conv2_2)
    conv2_3=Conv2D(128, (3, 1), activation='relu')(padd2_3)
    padd2_4=ZeroPadding2D((0,1))(conv2_3)
    conv2_4=Conv2D(128, (1, 3), activation='relu')(padd2_4)
    pool2=MaxPooling2D((2,2), strides=(2,2))(conv2_4)
    # group 3

    flat1=Flatten()(pool2)
    dens1=Dense(1024, activation='relu')(flat1)
    drop1=Dropout(0.5)(dens1)
    
    rnn=GRU(units=128,dropout=0.2,return_sequences=True)(input2)
    flatr=Flatten()(rnn)
    densr=Dense(256,activation='relu')(flatr)
    dropr=Dropout(0.5)(densr)

    concat=Concatenate(axis=1)([drop1,dropr])
    dens2=Dense(2048, activation='relu')(concat)
    drop2=Dropout(0.5)(dens2)


    out=Dense(2,activation='softmax')(drop2)

    if weights_path:
        model.load_weights(weights_path)

    return Model([input1,input2],out)

def rc():
    return "rc"
