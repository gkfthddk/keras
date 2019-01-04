from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout
from keras.layers import GRU, LSTM, SimpleRNN, Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np

def get_symbol(input_shape,weights_path=None):
    input2 = Input(shape=input_shape)

    rnn=GRU(units=64,dropout=0.2,return_sequences=True)(input2)
    flatr=Flatten()(rnn)
    densr=Dense(128,activation='relu')(flatr)
    dropr=Dropout(0.5)(densr)

    out=Dense(2,activation='softmax')(dropr)

    if weights_path:
        model.load_weights(weights_path)

    return Model(input2,out)

def rc():
    return "r"
