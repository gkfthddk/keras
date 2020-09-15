import numpy as np
from array import array
import tensorflow as tf
import ROOT as rt
import datetime
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0, 0, 0],
                                    [sinval, cosval, 0, 0, 0],
                                    [0, 0, 1, 0, 0],
                                    [0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 5)),rotation_matrix)
    return rotated_data
class DataGenerator(tf.keras.utils.Sequence):
    #def __init__(self,data, batch_size=32, num_classes=None, data_form="pixel",num_channel=None,num_point=2048,rotation=False,pix=90,target=0):
    def __init__(self,data,batch_size=32, num_classes=None, data_form="pixel",num_channel=None,num_point=2048,rotation=False,pix=90,target=0,stride=4,**kwargs):
        self.target=target
        self.stride=stride
        self.batch_size = batch_size
        self.num_classes = num_classes
        data_forms={"pixel":0,"voxel":1,"point":2}
        self.data_form=data_forms[data_form]
        num_channels=[4,2,5]
        self.default_channel=num_channels[self.data_form] 
        #self.signal=sigchain
        #self.background=bakchain
        #self.data=[self.signal,self.background]
        self.data=data
        #self.bak_total_len=self.background.GetEntries()
        #self.sig_total_len=self.signal.GetEntries()
        self.total_len=[sample.GetEntries() for sample in data]
        if(not num_classes):
          num_classes=len(data)
        self.num_classes=num_classes
        if(not num_channel):
          num_channel=self.default_channel
        self.num_channel=num_channel
        self.num_point=num_point
        self.pix=pix
        data_shapes={"pixel":(num_channel,pix,pix),"voxel":(num_channel,pix,pix,pix),"point":(num_point,num_channel)}
        self.data_shape=data_shapes[data_form]

        self.rotation=rotation
        self.on_epoch_end()

    def __len__(self):
        return int(sum(self.total_len) // self.batch_size)-1 # number of batches

    def __getitem__(self, index):
        #index = self.index[index * self.batch_size : (index + 1) * self.batch_size] # batch index list
        #batch = [self.indices[k] for k in index] indices[1] [1,2,3,4]
        X, y = self.__get_data(self.batch_size)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(sum(self.total_len)) # data index list
        self.choice_p=[1.*sample_len/sum(self.total_len) for sample_len in self.total_len]
        self.ent=[0]*self.num_classes
        self.test=False
        #if self.shuffle == True:
        #    np.random.shuffle(self.index)
    def GetEntry(pick,entry):
        self.data[pick].GetEntry(entry)
    def get_test(self):
        self.test=True
        Xout,Yout= self.__get_data(sum(self.total_len))
        self.on_epoch_end()
        return Xout, Yout

    def __get_data(self, batch_size):
        X=[]
        Y=[]
        pick=0
        num_data=0
        #now=datetime.datetime.now() 
        for i in range(batch_size):
          if(self.test==True):
            if(pick==self.num_classes):
              break
          else:
            pick=np.random.choice(self.num_classes,p=self.choice_p) # 0 background 1 signal
          num_data+=1
          self.data[pick].GetEntry(self.ent[pick])
          if(self.data_form==0):
            X.append([array("f",self.data[pick].image_ecor_s),array("f",self.data[pick].image_ecor_c),array("i",self.data[pick].image_n_s),array("i",self.data[pick].image_n_c)])
          if(self.data_form==1):
            X.append([array("f",self.data[pick].voxel_ecor_s),array("i",self.data[pick].voxel_n_s)])
          if(self.data_form==2):
            points=[]#phi eta depth s c 
            point_len=len(self.data[pick].fiber_depth)
            if(0):
              for j in range(self.num_point):
                if(j<point_len):
                  points.append([float(self.data[pick].fiber_phi[j]),float(self.data[pick].fiber_eta[j]),float(self.data[pick].fiber_depth[j]),float(self.data[pick].fiber_ecor[j]),float(bool(self.data[pick].fiber_iscerenkov[j]))])
                else:
                  points.append([0.]*self.default_channel)
            if(1):
              #points=np.array([np.array(self.data[pick].fiber_phi),np.array(self.data[pick].fiber_eta),np.array(self.data[pick].fiber_depth),np.array(self.data[pick].fiber_ecor),np.array(self.data[pick].fiber_iscerenkov,dtype="bool")]).transpose()
              points=np.array([np.array(self.data[pick].fiber_phi),np.array(self.data[pick].fiber_eta),np.array(self.data[pick].fiber_depth),np.array(self.data[pick].fiber_ecor_s),np.array(self.data[pick].fiber_ecor_c)]).transpose()
              points=sorted(points, key=lambda pnt:pnt[3],reverse=True)
              if(point_len<self.num_point):
                points=np.concatenate([points,np.zeros((self.num_point-point_len,self.default_channel))])
              else:
                points=points[:self.num_point]
               
            X.append(points)
          label=[0.]*self.num_classes
          label[pick]=1.
          if(self.target==0):Y.append(label)
          if(self.target==1):Y.append([self.data[pick].cmult,self.data[pick].nmult,self.data[pick].chad_mult,self.data[pick].nhad_mult])
          self.ent[pick]+=1
          if(self.ent[pick]==self.total_len[pick]):
            if(self.choice_p[pick]==1.):
              break

            if(self.test==True):
              pick=pick+1
            else:
              #self.choice_p[pick]=0.
              #self.choice_p[1-pick]=1.
              for k in range(len(self.choice_p)):
                if(k!=pick):
                  self.choice_p[k]+=self.choice_p[k]*self.choice_p[pick]/(1-self.choice_p[pick])
              self.choice_p[pick]=0.
              #print("!@#!@#",self.choice_p)
          
        if(self.data_form==0):
          Xout=np.array(X,dtype='float32').reshape((num_data,self.default_channel,self.pix,self.pix))
          Xout=Xout[:,:self.num_channel]
        if(self.data_form==1):
          Xout=np.array(X,dtype='float32').reshape((num_data,self.default_channel,self.pix,self.pix,self.pix))
          Xout=Xout[:,:self.num_channel]
        if(self.data_form==2):
          Xout=np.array(X,dtype='float32').reshape((num_data,self.num_point,self.default_channel))
          if(self.rotation==True):
            Xout=rotate_point_cloud(Xout)
          Xout=Xout[:,:,:self.num_channel]
        if(self.target==1):
          Y=np.array(Y,dtype='float32')
          Y=np.transpose(Y)
          Yout={"output{}".format(i) : Y[i] for i in self.stride}
          
        #print(datetime.datetime.now()-now)
        else:
          Yout=np.array(Y,dtype='float32')
        return Xout, Yout
import inspect
def prepare_data(data_path, num_file=500,tree_name="event",train_cut=0.7,val_cut=0.3,shuffle=False,**kwargs):
  #batch_size=32, num_classes=None,data_form="pixel",num_channel=None,num_point=2048,rotation=False,pix=23,target=0):
  trainchain=[]
  valchain=[]
  testchain=[]
  index=np.arange(num_file)
  if shuffle == True:
      np.random.shuffle(index)
  ent_train=int(train_cut*num_file*(1.-val_cut))
  ent_val=int(train_cut*num_file)
  ent_test=int(num_file)

  for i in range(len(data_path)):
    trainchain.append(rt.TChain(tree_name))
    for j in range(ent_train):
      trainchain[i].Add(data_path[i].format(index[j]))
    valchain.append(rt.TChain(tree_name))
    for j in range(ent_train,ent_val):
      valchain[i].Add(data_path[i].format(index[j]))
    testchain.append(rt.TChain(tree_name))
    for j in range(ent_val,ent_test):
      testchain[i].Add(data_path[i].format(index[j]))
  #print(prepare_data.__code__.co_varnames)
  #print(inspect.getargvalues(inspect.currentframe()))

  traindata=DataGenerator(trainchain, **kwargs)
  valdata=DataGenerator(valchain, **kwargs)
  testdata=DataGenerator(testchain, **kwargs)
  #traindata=DataGenerator(trainchain,batch_size=batch_size, num_classes=num_classes,data_form=data_form,num_channel=num_channel,num_point=num_point,rotation=rotation,pix=pix,target=target)
  #valdata=DataGenerator(valchain,batch_size=batch_size, num_classes=num_classes,data_form=data_form,num_channel=num_channel,num_point=num_point,rotation=rotation,pix=pix,target=target)
  #testdata=DataGenerator(testchain,batch_size=batch_size, num_classes=num_classes,data_form=data_form,num_channel=num_channel,num_point=num_point,rotation=rotation,pix=pix,target=target)
  return traindata, valdata, testdata


def mat_mul(A, B):
    return tf.linalg.matmul(A, B)
def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data

def tblock(g,channel,feat=64,loop=2,num_point=2048):
  x=g
  for i in range(loop):
    x=Convolution1D(feat,1,activation='relu',data_format='channels_last')(x)
    x=BatchNormalization()(x)
  x=MaxPooling1D(pool_size=num_point,data_format='channels_last')(x)
  for i in range(loop):
    x=Dense(feat,activation='relu')(x)
    x=BatchNormalization()(x)
  x=Dense(channel*channel,weights=[np.zeros([feat,channel*channel]),np.eye(channel).flatten().astype(np.float32)])(x)
  T=Reshape((channel,channel))(x)
  return mat_mul(g,T)

def pointmodel(num_point=2048,channel=4,num_classes=2,peak=0):
  
  # define optimizer
  adam = keras.optimizers.Adam(lr=0.001, decay=0.7)
  #adam = keras.optimizers.Adam()
  
  input_points = Input(shape=(num_point,channel ))#
  g=tblock(input_points,channel=4,feat=64,loop=1,num_point=num_point)
  chfeat=16
  g = Convolution1D(chfeat, 1, input_shape=(num_point, channel), activation='relu',data_format='channels_last')(g)#
  g = BatchNormalization()(g)#
  
  g=tblock(g,channel=chfeat,feat=64,loop=1,num_point=num_point)
  
  #g = Convolution1D(512, 1, activation='relu',data_format='channels_last')(g)
  #g = BatchNormalization()(g)
  #global_feature = Flatten()(MaxPooling1D(pool_size=num_point,data_format='channels_last')(g))#if use maxpooling many features needed
  g = Convolution1D(64, 1, activation='relu',data_format='channels_last')(g)
  g = BatchNormalization()(g)
  g = Convolution1D(2, 1, activation='relu',data_format='channels_first')(g)
  g = BatchNormalization()(g)
  global_feature = Flatten()(g)
  #g = Convolution1D(512, 1, activation='relu',data_format='channels_last')(g)
  #g = BatchNormalization()(g)
  
  # global_feature
  
  # point_net_cls
  #c = Dense(256, activation='relu')(global_feature)#
  #c = BatchNormalization()(c)#
  #c= Dropout(rate=0.7)(c)
  #c = Dense(256, activation="relu")(c)#2 validation increase
  #c = BatchNormalization()(c)#2 0.63
  #c= Dropout(rate=0.7)(c)
  #c = Dense(num_classes, activation='softmax',name="output1")(c)#
  if(peak==0):
    c = Dense(num_classes, activation='softmax',name="output1")(global_feature)#
  else:
    c = Dense(1, activation='linear',name="output1")(global_feature)#
  # --------------------------------------------------end of pointnet
  
  # print the model summary
  #model = Model(inputs=input_points, outputs=prediction)
  return Model(inputs=input_points, outputs=c)


if __name__== '__main__':
  #trainchain=[rt.TChain("event"),rt.TChain("event")]
  #for i in range(20):
  #  trainchain[0].Add("/pad/yulee/geant4/tester/analysis/fast/uJet50GeV_fastsim_{}.root".format(i))
  #  trainchain[1].Add("/pad/yulee/geant4/tester/analysis/fast/gJet50GeV_fastsim_{}.root".format(i))
  #a=DataGenerator(trainchain,data_form="point",batch_size=512)
  #b=a.__getitem__(10)
  #b=a.__getitem__(10)
  #b=a.__getitem__(10)
  path=["/pad/yulee/geant4/tester/analysis/fast/uJet50GeV_fastsim_{}.root","/pad/yulee/geant4/tester/analysis/fast/gJet50GeV_fastsim_{}.root"]
  train,val,test=prepare_data(path,data_form="pixel",batch_size=64,pix=90,num_channel=4,target=1)
  a=train.__getitem__(10)
  print(a[0].shape)
  c=a[1]
  print(c,c.shape)
  #train,val,test=prepare_data(path,data_form="voxel",batch_size=64,pix=90,num_channel=4)
  #a=train.__getitem__(10)
  #print(a[0].shape)
  #print(train.total_len,val.total_len,test.total_len)
  #x,y=test.get_test()
  #np.savez("pix90.npz",x=x,y=y)
  #train,val,test=prepare_data(path,data_form="point",batch_size=64,num_channel=4)
  #x,y=test.get_test()
  #np.savez("point.npz",x=x,y=y)
  #print(x.shape,y.shape)
  #print(test.__len__())
  #for i in range(test.__len__()):
  #  x,y=test.__getitem__(0)
