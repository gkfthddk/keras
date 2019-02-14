from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Concatenate, BatchNormalization, LeakyReLU
from keras.layers import GRU, LSTM, SimpleRNN, Input, Embedding, Lambda
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Conv1D, LocallyConnected1D, LocallyConnected2D
from keras.optimizers import SGD
import cv2, numpy as np

input_shape1=(10,33,33)
input_shape2=(20,9)
input_shape3=(20,5)
rnn={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp(input_shape2)-M0=ba(in0)-cv1(M0,32,1)-LeakyReLU(0.3)(M0)-gru(M0,256,0.2,1)-Flatten()(M0)-ba(M0)-dn(M0,256)-dr(M0)-dn(M0,256)-dr(M0)-out(M0)"}
ernn={"rc":"r","onehot":1,"num_input":1,"network":"in0=inp(input_shape3)-M0=emb(in0)-cv1(M0,32,1)-LeakyReLU(0.3)(M0)-gru(M0,256,0.2,1)-Flatten()(M0)-ba(M0)-dn(M0,512)-dr(M0)-dn(M0,512)-dr(M0)-out(M0)"}
rnn2={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp(input_shape2)-M0=ba(in0)-cv1(M0,64,1)-LeakyReLU(0.3)(M0)-gru(M0,512,0.2,1)-Flatten()(M0)-ba(M0)-dn(M0,1024)-dr(M0)-dn(M0,1024)-dr(M0)-out(M0)"}
rnn4={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp(input_shape2)-M0=ba(in0)-cv1(M0,64,1)-gru(M0,512,0.2,1)-Flatten()(M0)-ba(M0)-dn(M0,1024)-dr(M0)-dn(M0,1024)-dr(M0)-out(M0)"}
rnn3={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp(input_shape2)-M0=ba(in0)-cv1(M0,32,1)-gru(M0,256,0.2,1)-Flatten()(M0)-ba(M0)-dn(M0,512)-dr(M0)-dn(M0,512)-dr(M0)-out(M0)"}
lcnn={"rc":"c","onehot":0,"num_input":1,"network":"in0=inp(input_shape1)-M0=pad(in0)-lc2(M0,16)-pool(M0)-lc2(M0,32)-pool(M0)-Flatten()(M0)-dn(M0,2048)-dr(M0)-out(M0)"}
lcnn3={"rc":"c","onehot":0,"num_input":1,"network":"in0=inp(input_shape1)-M0=pad(in0)-lc2(M0,16)-pool(M0)-lc2(M0,32)-pool(M0)-Flatten()(M0)-dn(M0,2048)-dr(M0)-dn(M0,2048)-dr(M0)-out(M0)"}
cnn1={"rc":"c","onehot":0,"num_input":1,"network":"in0=inp(input_shape1)-M0=pad(in0)-cv2(M0,32)-pool(M0)-cv2(M0,64)-pool(M0)-Flatten()(M0)-dn(M0,2048)-dr(M0)-out(M0)"}
cnn12={"rc":"c","onehot":0,"num_input":1,"network":"in0=inp(input_shape1)-M0=ba(in0)-pad(M0)-cv2(M0,32)-pool(M0)-ba(M0)-cv2(M0,64)-pool(M0)-Flatten()(M0)-ba(M0)-dn(M0,2048)-dr(M0)-out(M0)"}
cnn11={"rc":"c","onehot":0,"num_input":1,"network":"in0=inp(input_shape1)-M0=pad(in0)-cv2(M0,32,(8,8))-pool(M0)-cv2(M0,32,(4,4))-pool(M0)-cv2(M0,32,(4,4))-pool(M0)-Flatten()(M0)-dn(M0,2048)-dr(M0)-out(M0)"}
cnn2={"rc":"c","onehot":0,"num_input":1,"network":"in0=inp(input_shape1)-M0=pad(in0)-cv2(M0,32)-pool(M0)-pad(M0)-cv2(M0,64)-pool(M0)-Flatten()(M0)-dn(M0,2048)-dr(M0)-out(M0)"}
cnn3={"rc":"c","onehot":0,"num_input":1,"network":"in0=inp(input_shape1)-M0=pad(in0)-cv2(M0,32)-pool(M0)-cv2(M0,64)-pool(M0)-Flatten()(M0)-dn(M0,2048)-dr(M0)-dn(M0,2048)-dr(M0)-out(M0)"}
cnn4={"rc":"c","onehot":0,"num_input":1,"network":"in0=inp(input_shape1)-M0=pad(in0)-cv2(M0,32)-ba(M0)-cv2(M0,32)-ba(M0)-pool(M0)-cv2(M0,64)-ba(M0)-cv2(M0,64)-ba(M0)-pool(M0)-Flatten()(M0)-dn(M0,2048)-dr(M0)-out(M0)"}
cnn5={"rc":"c","onehot":0,"num_input":1,"network":"in0=inp(input_shape1)-M0=pad(in0)-cv2(M0,32)-ba(M0)-cv2(M0,32)-ba(M0)-M1=pool(M0)-cv2(M1,64)-ba(M1)-cv2(M1,64)-ba(M1)-pool(M1)-Concatenate(axis=1)([Flatten()(M0),Flatten()(M1)])-dn(M1,2048)-dr(M1)-out(M1)"}

def inp(is2):
    return Input(shape=is2)
def ba(model):
    return BatchNormalization()(model)
def emb(model,dim=3):
    jet=Lambda(lambda x: x[:,:,:4])(model)
    pid=Lambda(lambda x: x[:,:,4:])(model)
    embed=Embedding(dim,5)(pid)
    reshape=Reshape((20,5))(embed)
    concat=Concatenate(axis=2)([jet,reshape])
    return BatchNormalization()(concat)
def pad(model,padding=(1,1)):
    return ZeroPadding2D(padding)(model)
def cv1(model,fil,ker,strid=1):
    return Conv1D(fil,ker,strides=strid)(model)
def cv2(model,fil,ker=(3,3),strid=(1,1)):
    return Conv2D(fil,ker,strides=strid)(model)
def lc2(model,fil,ker=(3,3),strid=(1,1)):
    return LocallyConnected2D(fil,ker,strides=strid)(model)
def pool(model,pool_size=(2,2),strid=(2,2)):
    return MaxPooling2D(pool_size,strid)(model)
def gru(model,units,drop,seq):
    return GRU(units,dropout=drop,return_sequences=seq)(model)
def dn(model,fil,act=None):
    return Dense(fil,activation=act)(model)
def dr(model,drop=0.5):
    return Dropout(drop)(model)
def out(model):
    return Dense(2,activation='softmax')(model)
    

def get_symbol(name):
    symbol=eval(name)
    net=symbol["network"].split("-")
    branch=[]
    inputs=[]
    idx=-1
    for i in range(len(net)):
        if(symbol["num_input"]>i):
            exec(net[i])
            inputs.append(eval(net[i][:3]))
        elif(net[i][2]=="="):
            idx+=1
            exec(net[i])
            branch.append(eval("M{}".format(idx)))
        else:
            exec("M{}=branch[{}]".format(idx,idx))
            branch[idx]=eval(net[i])
    if(symbol["num_input"]==0):return Model(inputs[0],branch[idx])
    else:return Model(inputs,branch[idx])

def onehot(name):
    symbol=eval(name)
    return symbol["onehot"]
