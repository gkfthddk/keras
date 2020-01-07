from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Concatenate, BatchNormalization, LeakyReLU, Activation, Permute
from keras.layers import GRU, LSTM, SimpleRNN, Input, Embedding, Lambda, GlobalAveragePooling2D, Softmax, Masking, Multiply
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Conv1D, LocallyConnected1D, LocallyConnected2D
from keras.layers import *
from keras.optimizers import SGD
import numpy as np

input_shape1=(10,33,33)
input_shape2=(20,9)
input_shape4=(30,9)
input_shape40=(40,9)
input_shape60=(60,9)
input_shape50=(50,9)
input_shape3=(20,5)
lstm2={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp(input_shape2)-M0=ba(in0)-cv1(M0,64,1)-LeakyReLU(0.3)(M0)-lst(M0,512,0.2,1)-Flatten()(M0)-ba(M0)-dn(M0,1024)-dr(M0)-dn(M0,1024)-dr(M0)-out(M0)"}
rnn={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp(input_shape2)-M0=ba(in0)-cv1(M0,64,1)-LeakyReLU(0.3)(M0)-gru(M0,512,0.2,1)-Flatten()(M0)-ba(M0)-dn(M0,256)-dr(M0)-dn(M0,256)-dr(M0)-out(M0)"}
rnna={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((None,9))-M0=ba(in0)-cv1(M0,64,1)-LeakyReLU(0.3)(M0)-gru(M0,512,0.2,1)-Flatten()(M0)-ba(M0)-dn(M0,256)-out(M0)"}
rnnb={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((None,9))-M0=ba(in0)-cv1(M0,64,1)-LeakyReLU(0.3)(M0)-gru(M0,256,0.2,1)-Flatten()(M0)-ba(M0)-dn(M0,128)-out(M0)"}
rnn3={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp(input_shape4)-M0=ba(in0)-cv1(M0,64,1)-LeakyReLU(0.3)(M0)-gru(M0,512,0.2,1)-Flatten()(M0)-ba(M0)-dn(M0,256)-dr(M0)-dn(M0,256)-dr(M0)-out(M0)"}
rnn00={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((None,9))-M0=Masking(mask_value=0.)(in0)-gru(M0,512,0.1,0)-out(M0)"}
rnn11={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((None,9))-M0=Masking(mask_value=0.)(in0)-gru(M0,256,0.1,0)-ba(M0)-dn(M0,256)-dn(M0,256)-out(M0)"}
rnn20={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((None,9))-M0=Masking(mask_value=0.)(in0)-gru(M0,512,0.1,0)-ba(M0)-dn(M0,256)-out(M0)"}
rnn21={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((None,9))-M0=Masking(mask_value=0.)(in0)-gru(M0,512,0.1,0)-ba(M0)-dn(M0,256)-ba(M0)-dn(M0,256)-out(M0)"}
rnn30={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((None,9))-M0=Masking(mask_value=0.)(in0)-gru(M0,256,0.,1)-ba(M0)-gru(M0,256,0.,0)-ba(M0)-dn(M0,256)-ba(M0)-out(M0)"}
rnn02={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((None,9))-M0=Masking(mask_value=0.)(in0)-gru(M0,512,0.,0)-ba(M0)-dn(M0,256)-ba(M0)-out(M0)"}
rnn31={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((None,9))-M0=Masking(mask_value=0.)(in0)-gru(M0,512,0.,1)-ba(M0)-gru(M0,512,0.,0)-ba(M0)-dn(M0,256)-ba(M0)-out(M0)"}
rnn33={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((None,9))-M0=Masking(mask_value=0.)(in0)-gru(M0,64,0.,1)-ba(M0)-gru(M0,128,0.,1)-ba(M0)-gru(M0,64,0.,1)-ba(M0)-gru(M0,16,0.,0)-out(M0)"}
rnn34={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((None,9))-M0=Masking(mask_value=0.)(in0)-gru(M0,64,0.,1)-ba(M0)-gru(M0,128,0.,1)-ba(M0)-gru(M0,64,0.,0)-ba(M0)-dn(M0,16)-out(M0)"}
rnnv10={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((None,9))-M0=gru(in0,256,0.1,0)-ba(M0)-dn(M0,256)-dr(M0)-out(M0)"}
rnnv11={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((None,9))-M0=gru(in0,256,0.1,0)-ba(M0)-dn(M0,256)-dr(M0)-dn(M0,256)-dr(M0)-out(M0)"}
rnnv20={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((None,9))-M0=gru(in0,512,0.1,0)-ba(M0)-dn(M0,256)-dr(M0)-out(M0)"}
rnnv21={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((None,9))-M0=gru(in0,512,0.1,0)-ba(M0)-dn(M0,256)-dr(M0)-dn(M0,256)-dr(M0)-out(M0)"}
rnn2={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp(input_shape2)-M0=cv1(in0,64,1)-LeakyReLU(0.3)(M0)-gru(M0,512,0.2,0)-dn(M0,256)-dr(M0)-dn(M0,256)-dr(M0)-out(M0)"}
rnn22={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((64,9))-M0=cv1(in0,64,1)-ba(M0)-gru(M0,256,0,0)-dn(M0,256)-dr(M0)-dn(M0,256)-dr(M0)-out(M0)"}
ernn={"rc":"r","onehot":1,"num_input":1,"network":"in0=inp(input_shape3)-M0=emb(in0)-cv1(M0,32,1)-LeakyReLU(0.3)(M0)-gru(M0,256,0.2,1)-Flatten()(M0)-ba(M0)-dn(M0,512)-dr(M0)-dn(M0,512)-dr(M0)-out(M0)"}
#worse with : no leakyrelu
cnn2={"rc":"c","onehot":0,"num_input":1,"network":"in0=inp(input_shape1)-M0=pad(in0)-cv2(M0,32)-pool(M0)-cv2(M0,64)-pool(M0)-Flatten()(M0)-dn(M0,128)-dn(M0,128)-dr(M0)-out(M0)"}
cnn11={"rc":"c","onehot":0,"num_input":1,"network":"in0=inp(input_shape1)-M0=pad(in0)-cv2(M0,32,(8,8))-pool(M0)-cv2(M0,32,(4,4))-pool(M0)-cv2(M0,32,(4,4))-pool(M0)-Flatten()(M0)-dn(M0,2048)-dr(M0)-out(M0)"}
#worse with : batchnorm block, localcnn 
cnn5={"rc":"c","onehot":0,"num_input":1,"network":"in0=inp(input_shape1)-M0=pad(in0)-cv2(M0,32)-ba(M0)-cv2(M0,32)-ba(M0)-M1=pool(M0)-cv2(M1,64)-ba(M1)-cv2(M1,64)-ba(M1)-pool(M1)-Concatenate(axis=1)([Flatten()(M0),Flatten()(M1)])-dn(M1,2048)-dr(M1)-out(M1)"}
cnn2={"rc":"c","onehot":0,"num_input":1,"network":"in0=inp(input_shape1)-M0=pad(in0)-ba(M0)-cv2(M0,32)-pool(M0)-ba(M0)-cv2(M0,64)-pool(M0)-Flatten()(M0)-ba(M0)-dn(M0,256)-ba(M0)-dn(M0,256)-out(M0)"}
bdtn={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((64,9))-M0=res(in0)-cv2(M0,128,(1,9),(1,1))-ba(M0)-Permute((3,2,1))(M0)-lc2(M0,1,(64,1),(1,1))-Flatten()(M0)-ba(M0)-out(M0)"}
nnn={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((64,9))-M0=res(in0)-cv2(M0,128,(1,9),(1,1))-ba(M0)-Permute((3,2,1))(M0)-lc2(M0,1,(64,1),(1,1))-Flatten()(M0)-ba(M0)-out(M0)"}
nnn3={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((64,9))-M0=res(in0)-cv2(M0,256,(1,9),(1,1))-ba(M0)-Permute((3,2,1))(M0)-lc2(M0,1,(64,1),(1,1))-Flatten()(M0)-ba(M0)-out(M0)"}
nnn2={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((64,9))-M0=res(in0)-cv2(M0,128,(1,9),(1,1))-ba(M0)-Permute((3,2,1))(M0)-lc2(M0,1,(64,1),(1,1))-Flatten()(M0)-ba(M0)-dn(M0,128)-ba(M0)-out(M0)"}
nnnseq={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((64,9))-M0=res(in0)-cv2(M0,128,(64,1),(1,1))-ba(M0)-Permute((2,3,1))(M0)-lc2(M0,1,(9,1),(1,1))-Flatten()(M0)-ba(M0)-out(M0)"}
jetnnn={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((2,64,9))-M0=res(div(in0,0))-cv2(M0,128,(1,9),(1,1))-ba(M0)-Permute((3,2,1))(M0)-cv2(M0,1,(64,1),(1,1))-M1=res(div(in0,1))-cv2(M1,128,(1,9),(1,1))-ba(M1)-Permute((3,2,1))(M1)-cv2(M1,1,(64,1),(1,1))-M2=Concatenate()([Flatten()(M0),Flatten()(M1)])-ba(M2)-out(M2,2)"}
jet1nnn={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((2,64,9))-M0=res(div(in0,0))-cv2(M0,128,(1,9),(1,1))-ba(M0)-Permute((3,2,1))(M0)-cv2(M0,1,(64,1),(1,1))-M1=Concatenate(axis=1)([Flatten()(M0),Flatten()(M0)])-ba(M1)-out(M1,2)"}
jetsnnn={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((2,64,9))-MMs=res(divs(in0,0))-cv2(M0,128,(1,9),(1,1))-ba(M0)-Permute((3,2,1))(M0)-cv2(M0,1,(64,1),(1,1))-M1=res(div(in0,1))-cv2(M1,128,(1,9),(1,1))-ba(M1)-Permute((3,2,1))(M1)-cv2(M1,1,(64,1),(1,1))-M2=Concatenate()([Flatten()(M0),Flatten()(M1)])-ba(M2)-out(M2,2)"}
#nnn={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((64,9))-M0=cv2(in0,64,(1,9),(1,0))-ba(M0)-cv2(M0,128,(64,1),(1,0))-ba(M0)-Flatten()(M0)-dn(M0,256)-out(M0)"}

#rcnn={"rc":"rc","onehot":0,"num_input":2,"network":"in0=inp(input_shape2)-M0=ba(in0)-cv1(M0,64,1)-LeakyReLU(0.3)(M0)-gru(M0,512,0.2,1)-Flatten()(M0)-ba(M0)-dn(M0,1024)-dr(M0)-dn(M0,1024)-dr(M0)-out(M0)"}


def res(model,ch=64):
    #check dimension!!
    return Reshape((-1,ch,9))(model)
def inp(is2):
    return Input(shape=is2)
def ba(model,axis=-1):
    return BatchNormalization(axis=axis)(model)
def emb(model,num_pid,length=128,dim=5):
    jet=Lambda(lambda x: x[:,:,:4])(model)
    pid=Lambda(lambda x: x[:,:,4])(model)
    embed=Embedding(num_pid+1,dim,mask_zero=True,input_length=length)(pid)
    #embed=Embedding(num_pid,dim,mask_zero=False,input_length=length)(pid)
    #reshape=Reshape((64,3))(embed)
    concat=Concatenate(axis=2)([jet,embed])
    #return BatchNormalization()(concat)
    return concat
def div(model,dim=0):
    one=Reshape((64,9))(Lambda(lambda x: x[:,dim,:,:])(model))
    return one 
def divs(model,stride):
    MMs=[]
    for i in range(stride):
      MMs.append(Lambda(lambda x: x[:,i,:,:])(model))
    return MMs
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
def lst(model,units,drop,seq):
    return LSTM(units,dropout=drop,return_sequences=seq)(model)
def dn(model,fil,act="relu"):
    return Dense(fil,activation=act)(model)
def dr(model,drop=0.5):
    return Dropout(drop)(model)
def out(model,stride=1):
    num=pow(2,stride)
    return Dense(num,activation='softmax')(model)
rnn21={"rc":"r","onehot":0,"num_input":1,"network":"in0=inp((None,9))-M0=Masking(mask_value=0.)(in0)-gru(M0,512,0.1,0)-ba(M0)-dn(M0,256)-ba(M0)-dn(M0,256)-out(M0)"}
def makepair(pair=[],stride=2,cand=[0,1]):
    for i in cand:
      if(stride==1):
        yield pair+[i]
      else:
        for loo in makepair(pair+[i],stride-1,cand):
          yield loo
def nnn(inp,stride):
    #MMs=divs(inp,stride)
    MMs=[]
    for i in range(stride):
      #MMs[i]=res(MMs[i])
      MMs.append(ba(inp[i]))#ba(MMs[i])
      MMs[i]=dn(MMs[i],128)
      MMs[i]=Permute((2,1))(MMs[i])
      MMs[i]=ba(MMs[i])
      MMs[i]=dn(MMs[i],128)
      MMs[i]=Permute((2,1))(MMs[i])
      MMs[i]=ba(MMs[i])
      MMs[i]=dn(MMs[i],32)
      MMs[i]=Permute((2,1))(MMs[i])
      MMs[i]=ba(MMs[i])
      MMs[i]=dn(MMs[i],32)
      MMs[i]=Flatten()(MMs[i])
      MMs[i]=ba(MMs[i])
      MMs[i]=dn(MMs[i],16)
      MMs[i]=ba(MMs[i])
      MMs[i]=Dense(2,activation='softmax',name="output{}".format(i+1))(MMs[i])
    return MMs
def nna3(inp,stride):
    MMs=[]
    mms=[]
    for i in range(stride):
      MMs.append(inp[i])
      MMs[i]=dn(MMs[i],8)
      MMs[i]=ba(MMs[i],1)
      MMs[i]=dn(MMs[i],16)
      MMs[i]=Permute((2,1))(MMs[i])
      MMs[i]=ba(MMs[i],1)
      MMs[i]=dn(MMs[i],64)
      MMs[i]=Permute((2,1))(MMs[i])#move to down?
      MMs[i]=ba(MMs[i],1)
      MMs[i]=Flatten()(MMs[i])
      MMs[i]=ba(MMs[i])
      MMs[i]=dn(MMs[i],64)
    for i in range(stride):
      mms.append(ba(Concatenate()([MMs[i],GaussianNoise(0.5)(MMs[i-1])])))
    for i in range(stride):
      MMs[i]=mms[i]
      MMs[i]=dn(MMs[i],16)
      MMs[i]=Dense(2,activation='softmax',name="output{}".format(i+1))(MMs[i])
      #MMs[i]=ba(MMs[i])
    return MMs
def pfr(inp,stride,seed):
    MMs=[]
    mm=Masking(mask_value=0.)(inp)
    mm=gru(mm,128,0,0)
    for i in range(stride):
      MMs.append(mm)
      MMs[i]=dn(MMs[i],64)
      MMs[i]=ba(MMs[i])
      MMs[i]=dn(MMs[i],64)
      MMs[i]=ba(MMs[i])
      MMs[i]=dn(MMs[i],16)
      MMs[i]=Dense(2,activation='softmax',name="output{}".format(i+1))(MMs[i])
      #MMs[i]=ba(MMs[i])
    return MMs
def pfk(inp,stride,seed):
    MMs=[]
    mm=dn(inp,32)
    mm=Permute((2,1))(mm)
    mm=ba(mm,1)
    mm=dn(mm,16)
    mm=Permute((2,1))(mm)#move to down?
    mm=ba(mm,1)
    mm=dn(mm,32)
    mm=Flatten()(mm)
    mm=ba(mm)
    mm=dn(mm,128)
    for i in range(stride):
      MMs.append(ba(mm))
      MMs[i]=dn(MMs[i],64)
      MMs[i]=Dense(2,activation='softmax',name="output{}".format(i+1))(MMs[i])
      #MMs[i]=ba(MMs[i])
    return MMs
def pfc(inp,stride,seed):
    mms=[]
    MMs=Reshape((4,55,72))(inp)
    for filters, kernels in zip([64, 32, 32],[4,2,1]):
      MMs=ba(MMs,1)
      MMs = Conv2D(filters=filters,
                   kernel_size=kernels,
                   strides=(1,1),
                   padding="valid",
                   activation="relu")(MMs)
      MMs=MaxPooling2D((3,3),(2,2))(MMs)
    MMs=Flatten()(MMs)
    MMs=ba(MMs)
    MMs=dn(MMs,128)
    MMs=ba(MMs)
    for i in range(stride):
      mms.append(MMs)
      #mms[i] = Conv2D(filters=2, kernel_size=1, strides=1)(mms[i])
      #mms[i] = GlobalAveragePooling2D()(mms[i])
      #mms[i] = Softmax(name="output{}".format(i+1))(mms[i])
      mms[i]=dn(mms[i],128)#128
      mms[i]=Dense(2,activation='softmax',name="output{}".format(i+1))(mms[i])
    return mms
def lck(inp,stride,seed):
    MMs=[]
    for i in range(stride):
      MMs.append(ba(res(inp[i])))#ba(MMs[i])
      MMs[i]=lc2(MMs[i],16,(1,9),(1,1))
      MMs[i]=Permute((3,2,1))(MMs[i])
      MMs[i]=ba(MMs[i])
      MMs[i]=lc2(MMs[i],64,(64,1),(1,1))
      MMs[i]=Flatten()(MMs[i])
      MMs[i]=ba(MMs[i])
      MMs[i]=dn(MMs[i],128)
    if(seed=="con"):incon=Concatenate()(MMs)
    if(seed=="add"):incon=Add()(MMs)
    if(seed=="sub"):incon=Subtract()(MMs)
    if(seed=="mul"):incon=Multiply()(MMs)
    if(seed=="ave"):incon=Average()(MMs)
    if(seed=="max"):incon=Maximum()(MMs)
    if(seed=="min"):incon=Minimum()(MMs)
    for i in range(stride):
      if(seed=="non"):MMs[i]=ba(MMs[i])
      else:MMs[i]=ba(incon)
      MMs[i]=dn(MMs[i],128)
      MMs[i]=dn(MMs[i],16)
      MMs[i]=Dense(2,activation='softmax',name="output{}".format(i+1))(MMs[i])
      #MMs[i]=ba(MMs[i])
    return MMs
def nnk(inp,stride,seed):
    MMs=[]
    for i in range(stride):
      MMs.append(inp[i])
      MMs[i]=ba(MMs[i],1)
      MMs[i]=dn(MMs[i],16)
      MMs[i]=Permute((2,1))(MMs[i])
      MMs[i]=ba(MMs[i],1)
      MMs[i]=dn(MMs[i],32)
      MMs[i]=Permute((2,1))(MMs[i])#move to down?
      MMs[i]=ba(MMs[i],1)
      MMs[i]=Flatten()(MMs[i])
      MMs[i]=ba(MMs[i])
      MMs[i]=dn(MMs[i],64)
      MMs[i]=ba(MMs[i])
    if(seed=="con"):incon=Concatenate()(MMs)
    if(seed=="add"):incon=Add()(MMs)
    if(seed=="sub"):incon=Subtract()(MMs)
    if(seed=="mul"):incon=Multiply()(MMs)
    if(seed=="ave"):incon=Average()(MMs)
    if(seed=="max"):incon=Maximum()(MMs)
    if(seed=="min"):incon=Minimum()(MMs)
    for i in range(stride):
      if(seed=="non"):MMs[i]=ba(MMs[i])
      else:MMs[i]=ba(incon)
      MMs[i]=dn(MMs[i],128)
      MMs[i]=dn(MMs[i],16)
      MMs[i]=Dense(2,activation='softmax',name="output{}".format(i+1))(MMs[i])
      #MMs[i]=ba(MMs[i])
    return MMs
def ernntest(inp,stride,seed,rsect=0,dsect=0):
    MMs=[]
    for i in range(stride):
      MMs.append(emb(inp[i],15,128,5))
      MMs[i]=Masking(mask_value=0.)(MMs[i])
      MMs[i]=ba(MMs[i])
      #MMs[i]=gru(MMs[i],64,0,1)
      if(rsect==0):
        MMs[i]=gru(MMs[i],128,0,0)
      if(rsect==1):
        MMs[i]=gru(MMs[i],256,0,0)
      if(rsect==2):
        MMs[i]=gru(MMs[i],512,0,0)
      if(rsect==3):
        MMs[i]=gru(MMs[i],128,0.1,0)
    #if(seed=="con"):incon=Concatenate()([Maximum()(MMs),Minimum()(MMs)])
    if(seed=="con"):incon=Concatenate()(MMs)
    if(seed=="add"):incon=Add()(MMs)
    if(seed=="sub"):incon=Subtract()(MMs)
    if(seed=="mul"):incon=Multiply()(MMs)
    if(seed=="ave"):incon=Average()(MMs)
    if(seed=="max"):incon=Maximum()(MMs)
    if(seed=="min"):incon=Minimum()(MMs)
    for i in range(stride):
      if(seed=="non"):MMs[i]=MMs[i]
      else:MMs[i]=incon
      if(dsect==0):
        MMs[i]=dn(MMs[i],128)#
      if(dsect==1):
        MMs[i]=dn(MMs[i],256)#
      if(dsect==2):
        MMs[i]=dn(MMs[i],512)#
      if(dsect==3):
        MMs[i]=dn(MMs[i],128)#
        MMs[i]=dn(MMs[i],128)#
      #MMs[i]=dn(MMs[i],16)
      MMs[i]=Dense(2,activation='softmax',name="output{}".format(i+1))(MMs[i])
    return MMs
def ernn(inp,stride,seed):
    MMs=[]
    for i in range(stride):
      MMs.append(emb(inp[i],15,128,5))
      MMs[i]=Masking(mask_value=0.)(MMs[i])
      MMs[i]=ba(MMs[i])
      #MMs[i]=gru(MMs[i],64,0,1)
      MMs[i]=gru(MMs[i],128,0,0)
      #MMs[i]=ba(MMs[i])
    #if(seed=="con"):incon=Concatenate()([Maximum()(MMs),Minimum()(MMs)])
    if(seed=="con"):incon=Concatenate()(MMs)
    if(seed=="add"):incon=Add()(MMs)
    if(seed=="sub"):incon=Subtract()(MMs)
    if(seed=="mul"):incon=Multiply()(MMs)
    if(seed=="ave"):incon=Average()(MMs)
    if(seed=="max"):incon=Maximum()(MMs)
    if(seed=="min"):incon=Minimum()(MMs)
    for i in range(stride):
      if(seed=="non"):MMs[i]=ba(MMs[i])
      else:MMs[i]=ba(incon)
      MMs[i]=dn(MMs[i],64)#
      #MMs[i]=dn(MMs[i],16)
      MMs[i]=Dense(2,activation='softmax',name="output{}".format(i+1))(MMs[i])
    return MMs
def ernn(inp,stride,seed):
    MMs=[]
    for i in range(stride):
      MMs.append(emb(inp[i],15,128,5))
      MMs[i]=Masking(mask_value=0.)(MMs[i])
      MMs[i]=ba(MMs[i])
      #MMs[i]=gru(MMs[i],64,0,1)
      MMs[i]=gru(MMs[i],128,0,0)
      #MMs[i]=ba(MMs[i])
    #if(seed=="con"):incon=Concatenate()([Maximum()(MMs),Minimum()(MMs)])
    if(seed=="con"):incon=Concatenate()(MMs)
    if(seed=="add"):incon=Add()(MMs)
    if(seed=="sub"):incon=Subtract()(MMs)
    if(seed=="mul"):incon=Multiply()(MMs)
    if(seed=="ave"):incon=Average()(MMs)
    if(seed=="max"):incon=Maximum()(MMs)
    if(seed=="min"):incon=Minimum()(MMs)
    for i in range(stride):
      if(seed=="non"):MMs[i]=ba(MMs[i])
      else:MMs[i]=ba(incon)
      MMs[i]=dn(MMs[i],64)#
      #MMs[i]=dn(MMs[i],16)
      MMs[i]=Dense(2,activation='softmax',name="output{}".format(i+1))(MMs[i])
    return MMs
def elstm(inp,stride,seed):
    MMs=[]
    for i in range(stride):
      MMs.append(emb(inp[i],15,128,5))
      MMs[i]=Masking(mask_value=0.)(MMs[i])
      MMs[i]=ba(MMs[i])
      #MMs[i]=gru(MMs[i],64,0,1)
      MMs[i]=lst(MMs[i],128,0,0)
      #MMs[i]=ba(MMs[i])
    #if(seed=="con"):incon=Concatenate()([Maximum()(MMs),Minimum()(MMs)])
    if(seed=="con"):incon=Concatenate()(MMs)
    if(seed=="add"):incon=Add()(MMs)
    if(seed=="sub"):incon=Subtract()(MMs)
    if(seed=="mul"):incon=Multiply()(MMs)
    if(seed=="ave"):incon=Average()(MMs)
    if(seed=="max"):incon=Maximum()(MMs)
    if(seed=="min"):incon=Minimum()(MMs)
    for i in range(stride):
      if(seed=="non"):MMs[i]=ba(MMs[i])
      else:MMs[i]=ba(incon)
      MMs[i]=dn(MMs[i],128)#
      MMs[i]=dn(MMs[i],128)#
      MMs[i]=dn(MMs[i],128)#
      #MMs[i]=dn(MMs[i],16)
      MMs[i]=Dense(2,activation='softmax',name="output{}".format(i+1))(MMs[i])
    return MMs
   
def rnn(inp,stride,seed):
    MMs=[]
    for i in range(stride):
      MMs.append(inp[i])
      MMs[i]=Masking(mask_value=0.)(MMs[i])
      MMs[i]=ba(MMs[i])
      #MMs[i]=gru(MMs[i],64,0,1)
      MMs[i]=gru(MMs[i],128,0,0)
      #MMs[i]=ba(MMs[i])
    #if(seed=="con"):incon=Concatenate()([Maximum()(MMs),Minimum()(MMs)])
    if(seed=="con"):incon=Concatenate()(MMs)
    if(seed=="add"):incon=Add()(MMs)
    if(seed=="sub"):incon=Subtract()(MMs)
    if(seed=="mul"):incon=Multiply()(MMs)
    if(seed=="ave"):incon=Average()(MMs)
    if(seed=="max"):incon=Maximum()(MMs)
    if(seed=="min"):incon=Minimum()(MMs)
    for i in range(stride):
      if(seed=="non"):MMs[i]=ba(MMs[i])
      else:MMs[i]=ba(incon)
      MMs[i]=dn(MMs[i],64)#
      #MMs[i]=dn(MMs[i],16)
      MMs[i]=Dense(2,activation='softmax',name="output{}".format(i+1))(MMs[i])
    return MMs

def nnt(inp,stride,seed):
    MMs=[]
    for i in range(stride):
      MMs.append(inp[i])
      MMs[i]=dn(MMs[i],8)
      MMs[i]=ba(MMs[i],1)
      MMs[i]=dn(MMs[i],16)
      MMs[i]=Permute((2,1))(MMs[i])
      MMs[i]=ba(MMs[i],1)
      MMs[i]=dn(MMs[i],64)
      MMs[i]=Permute((2,1))(MMs[i])#move to down?
      MMs[i]=ba(MMs[i],1)
      MMs[i]=Flatten()(MMs[i])
      MMs[i]=ba(MMs[i])
      MMs[i]=dn(MMs[i],64)
    if(seed=="con"):incon=Concatenate()(MMs)
    if(seed=="add"):incon=Add()(MMs)
    if(seed=="sub"):incon=Subtract()(MMs)
    if(seed=="mul"):incon=Multiply()(MMs)
    if(seed=="ave"):incon=Average()(MMs)
    if(seed=="max"):incon=Maximum()(MMs)
    if(seed=="min"):incon=Minimum()(MMs)
    for i in range(stride):
      if(seed=="non"):MMs[i]=ba(MMs[i])
      else:MMs[i]=ba(incon)
      MMs[i]=dn(MMs[i],16)
    if(seed=="con"):incon=Concatenate()(MMs)
    if(seed=="add"):incon=Add()(MMs)
    if(seed=="sub"):incon=Subtract()(MMs)
    if(seed=="mul"):incon=Multiply()(MMs)
    if(seed=="ave"):incon=Average()(MMs)
    if(seed=="max"):incon=Maximum()(MMs)
    if(seed=="min"):incon=Minimum()(MMs)
    for i in range(stride):
      if(seed=="non"):MMs[i]=ba(MMs[i])
      else:MMs[i]=ba(incon)
      MMs[i]=dn(MMs[i],16)
      MMs[i]=Dense(2,activation='softmax',name="output{}".format(i+1))(MMs[i])
      #MMs[i]=ba(MMs[i])
    return MMs
def nnc(inp,stride):
    MMs=[]
    incon=Concatenate(axis=2)(inp)
    incon=dn(incon,32)
    for i in range(stride):
      MMs.append(dn(incon,128))
      MMs[i]=Permute((2,1))(MMs[i])
      MMs[i]=ba(MMs[i],1)
      MMs[i]=dn(MMs[i],128)
      MMs[i]=Permute((2,1))(MMs[i])
      #MMs[i]=ba(MMs[i])
      #MMs[i]=dn(MMs[i],32)
      #MMs[i]=Permute((2,1))(MMs[i])
      #MMs[i]=ba(MMs[i])
      #MMs[i]=dn(MMs[i],32)
      MMs[i]=Flatten()(MMs[i])
      MMs[i]=ba(MMs[i])
      MMs[i]=dn(MMs[i],64)
      MMs[i]=ba(MMs[i])
      MMs[i]=Dense(2,activation='softmax',name="output{}".format(i+1))(MMs[i])
      #MMs[i]=ba(MMs[i])
    return MMs
def nnm(inp,stride,seed):
    MMs=[]
    for i in range(stride):
      MMs.append(inp[i])
      MMs[i]=dn(MMs[i],8)
      MMs[i]=ba(MMs[i],1)
    if(seed=="con"):incon=Concatenate(axis=2)(MMs)
    if(seed=="add"):incon=Add()(MMs)
    if(seed=="sub"):incon=Subtract()(MMs)
    if(seed=="mul"):incon=Multiply()(MMs)
    if(seed=="ave"):incon=Average()(MMs)
    if(seed=="max"):incon=Maximum()(MMs)
    if(seed=="min"):incon=Minimum()(MMs)
    if(seed=="dot"):incon=Dot()(MMs)
    for i in range(stride):
      MMs[i]=ba(incon,1)
      MMs[i]=dn(MMs[i],16)
      MMs[i]=Permute((2,1))(MMs[i])
      MMs[i]=ba(MMs[i],1)
      MMs[i]=dn(MMs[i],64)
      MMs[i]=Permute((2,1))(MMs[i])
      #MMs[i]=ba(MMs[i])
      #MMs[i]=dn(MMs[i],32)
      #MMs[i]=Permute((2,1))(MMs[i])
      #MMs[i]=ba(MMs[i])
      #MMs[i]=dn(MMs[i],32)
      MMs[i]=Flatten()(MMs[i])
      MMs[i]=ba(MMs[i])
      MMs[i]=dn(MMs[i],64)
      MMs[i]=ba(MMs[i])
      MMs[i]=Dense(2,activation='softmax',name="output{}".format(i+1))(MMs[i])
      #MMs[i]=ba(MMs[i])
    return MMs
def nnb(inp,stride,seed):
    MMs=[]
    for i in range(stride):
      MMs.append(inp[i-1])
      MMs[i]=dn(MMs[i],16)
      MMs[i]=Permute((2,1))(MMs[i])
      MMs[i]=ba(MMs[i],1)
      MMs[i]=dn(MMs[i],64)
      MMs[i]=Permute((2,1))(MMs[i])
      #MMs[i]=ba(MMs[i])
      #MMs[i]=dn(MMs[i],32)
      #MMs[i]=Permute((2,1))(MMs[i])
      #MMs[i]=ba(MMs[i])
      #MMs[i]=dn(MMs[i],32)
      MMs[i]=Flatten()(MMs[i])
      MMs[i]=ba(MMs[i])
      MMs[i]=dn(MMs[i],64)
      MMs[i]=ba(MMs[i])
      MMs[i]=Dense(2,activation='softmax',name="output{}".format(i+1))(MMs[i])
      #MMs[i]=ba(MMs[i])
    return MMs
def nncode(inp,stride):
    MMs=[]
    for i in range(stride):
      MMs.append(ba(inp[i],1))#ba(MMs[i])
      MMs[i]=dn(MMs[i],128)
      #MMs[i]=Permute((2,1))(MMs[i])
      #MMs[i]=dn(MMs[i],128)
      #MMs[i]=Permute((2,1))(MMs[i])
      MMs[i]=dn(MMs[i],32)
      #MMs[i]=Permute((2,1))(MMs[i])
      #MMs[i]=dn(MMs[i],32)
      #MMs[i]=Flatten()(MMs[i])
      MMs[i]=dn(MMs[i],16)

      MMs[i]=dn(MMs[i],32)
      MMs[i]=dn(MMs[i],128)
      MMs[i]=Dense(32,activation="sigmoid")(MMs[i])

      #MMs[i]=dn(MMs[i],32*32)
      #MMs[i]=Reshape((32,32))(MMs[i])
      #MMs[i]=dn(MMs[i],128)
      #MMs[i]=Permute((2,1))(MMs[i])
      #MMs[i]=dn(MMs[i],128)
      #MMs[i]=Permute((2,1))(MMs[i])
      #MMs[i]=dn(MMs[i],32)
      #MMs[i]=Permute((2,1))(MMs[i])
      #MMs[i]=Dense(1,activation="sigmoid")(MMs[i])
      #MMs[i]=Dense(2,activation='softmax',name="output{}".format(i+1))(MMs[i])
    return MMs
def nnn2mod(inp,stride,mod):
    MMs=[]
    for i in range(stride):
      MMs.append(ba(res(inp[i],1)))#ba(MMs[i])
      MMs[i]=lc2(MMs[i],128,(1,9),(1,1))
      MMs[i]=Permute((3,2,1))(MMs[i])
      MMs[i]=ba(MMs[i],1)
      MMs[i]=lc2(MMs[i],1,(64,1),(1,1))
      MMs[i]=Flatten()(MMs[i])
      #MMs[i]=ba(MMs[i])
    mss=[]
    for i in range(stride):
      if(stride==1):mss.append(MMs[0])
      else:mss.append(Concatenate()([MMs[l] for l in range(len(MMs))]))
      #else:mss.append(Concatenate()([MMs[l] if l!=i else dn(MMs[l],128) for l in range(len(MMs))]))
      mss[i]=ba(mss[i])
      mss[i]=dn(mss[i],128)
      mss[i]=ba(mss[i])
      #if(mod==1):mss[i]=out(mss[i],1)
      if(mod==1):
        mss[i]=out(mss[i],1)
        mss[i]=[Lambda(lambda x: x[:,0:1])(mss[i]),Lambda(lambda x: x[:,1:2])(mss[i])]
    return mss
def pfcon(name,stride,seed=""):
    if("c" in name):
      inp=Input(shape=(4,55*72))
    else:
      inp=Input(shape=(128,4))
    if(name=="pfk"):
      MMs=pfk(inp,stride,seed)
    if(name=="pfr"):
      MMs=pfr(inp,stride,seed)
    if(name=="pfc"):
      MMs=pfc(inp,stride,seed)
    return Model(inp,MMs)
def jetcon(name,stride,seed="",rsect=0,dsect=0):
    inp=[]
    for i in range(stride):
      if("code" in name):
        inp.append(Input(shape=(64,)))
      else:
        #inp.append(Input(shape=(64,9)))
        inp.append(Input(shape=(128,5)))
    if(name=="rnn"):
      MMs=rnn(inp,stride,seed)
    if(name=="ernn"):
      MMs=ernn(inp,stride,seed)
    if(name=="ernn2"):
      MMs=ernn2(inp,stride,seed)
    if(name=="ernntest"):
      MMs=ernntest(inp,stride,seed,rsect,dsect)
    if(name=="elstm"):
      MMs=elstm(inp,stride,seed)
    if(name=="nnn"):
      MMs=nnn(inp,stride)
    if(name=="lck"):
      MMs=lck(inp,stride,seed)
    if(name=="nna3"):
      MMs=nna3(inp,stride)
    if(name=="nnc"):
      MMs=nnc(inp,stride)
    if(name=="nnm"):
      MMs=nnm(inp,stride,seed)
    if(name=="nnb"):
      MMs=nnb(inp,stride,seed)
    if(name=="nnk"):
      MMs=nnk(inp,stride,seed)
    if(name=="nncode"):
      MMs=nncode(inp,stride)
    return Model(inp,MMs)
def jetconmod(name,stride,mod):
    inp=[]
    for i in range(stride):
      inp.append(Input(shape=(64,9)))
    if(name=="nnn2" and mod==1):
      MMs=nnn2mod(inp,stride,mod)
      #if(len(MMs)>1):model=Concatenate()(MMs)
      LAS=[]
      for cand in list(makepair(stride=stride)):
        las=[]
        for i in range(len(MMs)):
          las.append(MMs[i][cand[i]])
        LAS.append(las)
      if(len(LAS[0])<2):
        out=Concatenate()([las[0] for las in LAS])
      else:
        out=Concatenate(name="output")([Multiply()(las) for las in LAS])
    if(name=="nnn2" and mod==2):
      MMs=nnn2mod(inp,stride,mod)
      model=Concatenate()(MMs)
      out=Dense(pow(2,stride),activation='softmax',name="output")(model)
    return Model(inp,out)
def conv_block(tensor,
               filters,
               kernel_size=5,
               strides=(1, 1),
               activation="relu",
               padding="valid"):
    tensor = Conv2D(filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding)(tensor)
    tensor = BatchNormalization(axis=1)(tensor)
    tensor = Activation(activation)(tensor)
    return tensor
def jetcnn(stride=2,seed="con",x_shape=(10, 33*33)):
    inp=[]
    MMs=[]
    mms=[]
    for i in range(stride):
      inp.append(Input(shape=x_shape))
      MMs.append(Reshape((10,33,33))(inp[i]))
      for filters, kernels in zip([64, 32],[3,3]):
        MMs[i]=ba(MMs[i])
        MMs[i] = Conv2D(filters=filters,
                        kernel_size=kernels,
                        strides=(1,1),
                        padding="valid",
                        activation="relu")(MMs[i])
        MMs[i]=MaxPooling2D((2,2),(2,2))(MMs[i])
      MMs[i] = Conv2D(filters=128, kernel_size=1, strides=1)(MMs[i])
      MMs[i] = GlobalAveragePooling2D()(MMs[i])
      #MMs[i]=Flatten()(MMs[i])
      #MMs[i]=ba(MMs[i])
      #MMs[i]=dn(MMs[i],64)
    if(stride==1):
      mms.append(MMs[0])
    else:
      if(seed=="con"):incon=Concatenate()(MMs)
      if(seed=="min"):incon=Minimum()(MMs)
      if(seed=="max"):incon=Maximum()(MMs)
    """  for i in range(stride):
        if(seed=="non"):mms.append(MMs[i])
        else:mms.append(incon)

    for i in range(stride):
      #mms[i]=ba(mms[i])
      #mms[i]=dn(mms[i],128)#128
      if(seed!="non"):
        if(seed=="con"):mms[i]=Dense(256,activation='softmax')(mms[i])#
        else:mms[i]=Dense(128,activation='softmax')(mms[i])#
    for i in range(stride):#
      if(seed=="non"):
        MMs[i]=mms[i]
      else:
        MMs[i]=Multiply()([mms[stride-i-1],incon])#"""
    for i in range(stride):#
      if(seed=="non"):mms.append(MMs[i])
      else:mms.append(incon)
      #mms[i]=dr(mms[i],0.8)
      mms[i]=ba(mms[i])
      mms[i]=dn(mms[i],64)
      #mms[i]=dn(mms[i],16)
      mms[i]=Dense(2,activation='softmax',name="output{}".format(i+1))(mms[i])
    return Model(inp,mms)
def jetcv(stride=2,seed="con",x_shape=(10, 33*33)):
    inp=[]
    MMs=[]
    mms=[]
    for i in range(stride):
      inp.append(Input(x_shape))
      MMs.append(Reshape((10,33,33))(inp[i]))
      for filters in [32, 64, 64, 32]:
        MMs[i]=ba(MMs[i],1)
        MMs[i] = Conv2D(filters=filters,
                        kernel_size=5,
                        strides=(1,1),
                        padding="valid",
                        activation="relu")(MMs[i])
    if(stride==1):
      mms.append(MMs[0])
    else:
      incon=Concatenate(axis=1)(MMs)
      for i in range(stride):
        if(seed=="non"):mms.append(MMs[i])
        else:mms.append(incon)

    for i in range(stride):
      mms[i]=ba(mms[i],1)
      mms[i] = Conv2D(filters=2, kernel_size=1, strides=1)(mms[i])
      mms[i] = GlobalAveragePooling2D()(mms[i])
      mms[i] = Softmax(name="output{}".format(i+1))(mms[i])
    return Model(inp,mms)

def modelss(x_shape=(10, 33*33)):
    x = Input(x_shape)
    h=x
    #h=Reshape((10,33,33))(x)

    for filters in [32, 64, 64, 32]:
        h = conv_block(h, filters, activation="relu")
    h = Conv2D(filters=2, kernel_size=1, strides=1)(h)
    logits = GlobalAveragePooling2D()(h)
    y_score = Softmax(name="output1")(logits)
    model = Model(inputs=x, outputs=y_score)
    return model



def get_symbol(name,stride):
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
