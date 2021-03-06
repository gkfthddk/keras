from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, LeakyReLU
from keras.layers import GRU, LSTM, SimpleRNN, Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Conv1D
from keras.optimizers import SGD
import cv2, numpy as np

def get_symbol(input_shape,weights_path=None):
    input2 = Input(shape=input_shape)
    batch1 = BatchNormalization()(input2)
    conv1=Conv1D(32,1,strides=1)(batch1)
    relu=LeakyReLU(alpha=0.2)(conv1)
    rnn=GRU(units=256,dropout=0.2,return_sequences=True)(relu)
    flatr=Flatten()(rnn)
    batch2=BatchNormalization()(flatr)
    dens1=Dense(256,activation='relu')(batch2)
    drop1=Dropout(0.5)(dens1)
    dens2=Dense(256,activation='relu')(drop1)
    drop2=Dropout(0.5)(dens2)

    out=Dense(2,activation='softmax')(drop2)

    if weights_path:
        model.load_weights(weights_path)

    return Model(input2,out)

def rc():
    return "r"
