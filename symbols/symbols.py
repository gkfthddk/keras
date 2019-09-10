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
    return Reshape((-1,ch,9))(model)
def inp(is2):
    return Input(shape=is2)
def ba(model):
    return BatchNormalization()(model)
def emb(model,dim=3):
    jet=Lambda(lambda x: x[:,:,:4])(model)
    pid=Lambda(lambda x: x[:,:,4:])(model)
    embed=Embedding(5,dim)(pid)
    reshape=Reshape((20,3))(embed)
    concat=Concatenate(axis=2)([jet,reshape])
    return BatchNormalization()(concat)
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
def dn(model,fil,act=None):
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
def nnn1(inp,stride):
    MMs=[]
    for i in range(stride):
      MMs.append(ba(res(inp[i])))#ba(MMs[i])
      MMs[i]=lc2(MMs[i],128,(1,9),(1,1))
      MMs[i]=Permute((3,2,1))(MMs[i])
      MMs[i]=ba(MMs[i])
      MMs[i]=lc2(MMs[i],1,(64,1),(1,1))
      MMs[i]=Flatten()(MMs[i])
      #MMs[i]=ba(MMs[i])
    mss=[]
    for i in range(stride):
      if(stride==1):mss.append(MMs[0])
      else:mss.append(Concatenate()([ms for ms in MMs]))
      mss[i]=ba(mss[i])
      mss[i]=Dense(2,activation='softmax',name="output{}".format(i+1))(mss[i])
    return mss
def nnn2(inp,stride):
    MMs=[]
    for i in range(stride):
      MMs.append(ba(res(inp[i])))#ba(MMs[i])
      MMs[i]=lc2(MMs[i],128,(1,9),(1,1))
      MMs[i]=Permute((3,2,1))(MMs[i])
      MMs[i]=ba(MMs[i])
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
      mss[i]=Dense(2,activation='softmax',name="output{}".format(i+1))(mss[i])
      #mss[i]=out(mss[i],1)
    #  #mss[i]=[Lambda(lambda x: x[:,0:1])(mss[i]),Lambda(lambda x: x[:,1:2])(mss[i])]
    return mss
def nnn3(inp,stride):
    MMs=[]
    for i in range(stride):
      MMs.append(ba(res(inp[i])))#ba(MMs[i])
      MMs[i]=lc2(MMs[i],64,(1,9),(1,1))
      MMs[i]=Permute((3,2,1))(MMs[i])
      MMs[i]=ba(MMs[i])
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
      mss[i]=dn(mss[i],128)
      mss[i]=ba(mss[i])
      mss[i]=Dense(2,activation='softmax',name="output{}".format(i+1))(mss[i])
      #mss[i]=out(mss[i],1)
    #  #mss[i]=[Lambda(lambda x: x[:,0:1])(mss[i]),Lambda(lambda x: x[:,1:2])(mss[i])]
    return mss
def nna3(inp,stride):
    MMs=[]
    for i in range(stride):
      MMs.append(ba(res(inp[i])))#ba(MMs[i])
      MMs[i]=lc2(MMs[i],64,(1,9),(1,1))
      MMs[i]=Permute((3,2,1))(MMs[i])
      MMs[i]=ba(MMs[i])
      MMs[i]=lc2(MMs[i],1,(64,1),(1,1))
      MMs[i]=Flatten()(MMs[i])
      #MMs[i]=ba(MMs[i])
    mss=[]
    for i in range(stride):
      if(stride==1):mss.append(MMs[0])
      else:mss.append(Add()([MMs[l] for l in range(len(MMs))]))
      #else:mss.append(Concatenate()([MMs[l] if l!=i else dn(MMs[l],128) for l in range(len(MMs))]))
      mss[i]=ba(mss[i])
      mss[i]=dn(mss[i],128)
      mss[i]=ba(mss[i])
      mss[i]=Dense(2,activation='softmax',name="output{}".format(i+1))(mss[i])
      #mss[i]=out(mss[i],1)
    #  #mss[i]=[Lambda(lambda x: x[:,0:1])(mss[i]),Lambda(lambda x: x[:,1:2])(mss[i])]
    return mss
def nnc(inp,stride):
    MMs=[]
    incon=Concatenate(axis=2)(inp)
    for i in range(stride):
      MMs.append(dn(incon,128))
      MMs[i]=Permute((2,1))(MMs[i])
      MMs[i]=ba(MMs[i])
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
def nnm(inp,stride):
    MMs=[]
    #incon=Concatenate(axis=2)(inp)
    incon=Maximum()(inp)
    for i in range(stride):
      MMs.append(dn(incon,128))
      MMs[i]=Permute((2,1))(MMs[i])
      MMs[i]=ba(MMs[i])
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
def nncode(inp,stride):
    MMs=[]
    for i in range(stride):
      MMs.append(ba(inp[i]))#ba(MMs[i])
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
      MMs.append(ba(res(inp[i])))#ba(MMs[i])
      MMs[i]=lc2(MMs[i],128,(1,9),(1,1))
      MMs[i]=Permute((3,2,1))(MMs[i])
      MMs[i]=ba(MMs[i])
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
def rnn(inp,stride):
    MMs=divs(inp,stride)
    for i in range(stride):
      MMs[i]=Reshape((64,9))(MMs[i])
      MMs[i]=ba(MMs[i])
      MMs[i]=gru(MMs[i],256,0,0)
      MMs[i]=ba(MMs[i])
      MMs[i]=dn(MMs[i],128)
      MMs[i]=ba(MMs[i])
      MMs[i]=dn(MMs[i],128)
      MMs[i]=ba(MMs[i])
      MMs[i]=out(MMs[i],1)
      MMs[i]=[Lambda(lambda x: x[:,0:1])(MMs[i]),Lambda(lambda x: x[:,1:2])(MMs[i])]
    return MMs
def jetcon(name,stride):
    inp=[]
    for i in range(stride):
      if("code" in name):
        inp.append(Input(shape=(32,)))
      else:
        inp.append(Input(shape=(32,4)))
        #inp.append(Input(shape=(64,9)))
    if(name=="rnn"):
      MMs=rnn(inp,stride)
    if(name=="nnn"):
      MMs=nnn(inp,stride)
    if(name=="nnn2"):
      MMs=nnn2(inp,stride)
    if(name=="nnn3"):
      MMs=nnn3(inp,stride)
    if(name=="nna3"):
      MMs=nna3(inp,stride)
    if(name=="nnc"):
      MMs=nnc(inp,stride)
    if(name=="nnm"):
      MMs=nnm(inp,stride)
    if(name=="nnn1"):
      MMs=nnn1(inp,stride)
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

def modelss(x_shape=(10, 33, 33)):
    x = Input(x_shape)

    h = x
    for filters in [32, 64, 64, 32]:
        h = conv_block(h, filters, activation="relu")
    h = Conv2D(filters=2, kernel_size=1, strides=1)(h)
    logits = GlobalAveragePooling2D()(h)
    y_score = Softmax()(logits)
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
