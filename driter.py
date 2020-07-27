import numpy as np
import tensorflow as tf
import ROOT as rt
import datetime

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
    def __init__(self,data, batch_size=32, num_classes=None, data_form="pixel",num_channel=None,rotation=False):
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
        data_shapes={"pixel":(num_channel,23,23),"voxel":(num_channel,23,23,23),"point":(2048,num_channel)}
        self.data_shape=data_shapes[data_form]

        self.rotation=rotation
        self.on_epoch_end()

    def __len__(self):
        return int(sum(self.total_len) // self.batch_size) # number of batches

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
    def get_test(self):
        self.test=True
        Xout,Yout= self.__get_data(sum(self.total_len))
        self.on_epoch_end()
        return Xout, Yout

    def __get_data(self, batch_size):
        X=[]
        Y=[]
        pick=0
        now=datetime.datetime.now() 
        for i in xrange(batch_size):
          if(self.test==True):
            if(pick==self.num_classes):
              break
          else:
            pick=np.random.choice(self.num_classes,p=self.choice_p) # 0 background 1 signal
          self.data[pick].GetEntry(self.ent[pick])
          if(self.data_form==0):
            X.append([self.data[pick].image_ecor_s,self.data[pick].image_ecor_c,self.data[pick].image_n_s,self.data[pick].image_n_c])
          if(self.data_form==1):
            X.append([self.data[pick].voxel_ecor_s,self.data[pick].voxel_n_s])
          if(self.data_form==2):
            points=[]#phi eta depth s c 
            point_len=len(self.data[pick].fiber_depth)
            if(0):
              for j in range(2048):
                if(j<point_len):
                  points.append([float(self.data[pick].fiber_phi[j]),float(self.data[pick].fiber_eta[j]),float(self.data[pick].fiber_depth[j]),float(self.data[pick].fiber_ecor[j]),float(bool(self.data[pick].fiber_iscerenkov[j]))])
                else:
                  points.append([0.]*self.default_channel)
            if(1):
              #points=np.array([np.array(self.data[pick].fiber_phi),np.array(self.data[pick].fiber_eta),np.array(self.data[pick].fiber_depth),np.array(self.data[pick].fiber_ecor),np.array(self.data[pick].fiber_iscerenkov,dtype="bool")]).transpose()
              points=np.array([np.array(self.data[pick].fiber_phi),np.array(self.data[pick].fiber_eta),np.array(self.data[pick].fiber_depth),np.array(self.data[pick].fiber_ecor_s),np.array(self.data[pick].fiber_ecor_c)]).transpose()
              if(point_len<2048):
                points=np.concatenate([points,np.zeros((2048-point_len,self.default_channel))])
              else:
                points=points[:2048]
               
            X.append(points)
          label=[0.]*self.num_classes
          label[pick]=1.
          Y.append(label)
          self.ent[pick]+=1
          if(self.ent[pick]==self.total_len[pick]):
            if(self.test==True):
              pick=pick+1
            else:
              #self.choice_p[pick]=0.
              #self.choice_p[1-pick]=1.
              for k in range(len(self.choice_p)):
                if(k!=pick):
                  self.choice_p[k]+=self.choice_p[k]*self.choice_p[pick]/(1-self.choice_p[pick])
              self.choice_p[pick]=0.
          
        if(self.data_form==0):
          Xout=np.array(X,dtype='float32').reshape((batch_size,self.default_channel,23,23))
          Xout=Xout[:,:self.num_channel]
        if(self.data_form==1):
          Xout=np.array(X,dtype='float32').reshape((batch_size,self.default_channel,23,23,23))
          Xout=Xout[:,:self.num_channel]
        if(self.data_form==2):
          Xout=np.array(X,dtype='float32').reshape((batch_size,2048,self.default_channel))
          if(self.rotation==True):
            Xout=rotate_point_cloud(Xout)
          Xout=Xout[:,:,:self.num_channel]
        Yout=np.array(Y,dtype='float32')
        print(datetime.datetime.now()-now)
        return Xout, Yout
import inspect
def prepare_data(data_path, num_files=500,tree_name="event",train_cut=0.7,val_cut=0.3,shuffle=False,   batch_size=32, num_classes=None,data_form="pixel",num_channel=None,rotation=False):
  trainchain=[]
  valchain=[]
  testchain=[]
  index=np.arange(num_files)
  if shuffle == True:
      np.random.shuffle(index)
  ent_train=int(train_cut*num_files*(1.-val_cut))
  ent_val=int(train_cut*num_files)
  ent_test=int(num_files)

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

  traindata=DataGenerator(trainchain,batch_size=batch_size, num_classes=num_classes,data_form=data_form,num_channel=num_channel,rotation=rotation)
  valdata=DataGenerator(valchain,batch_size=batch_size, num_classes=num_classes,data_form=data_form,num_channel=num_channel,rotation=rotation)
  testdata=DataGenerator(testchain,batch_size=batch_size, num_classes=num_classes,data_form=data_form,num_channel=num_channel,rotation=rotation)
  return traindata, valdata, testdata


if __name__== '__main__':
  #sigchain=rt.TChain("event")
  #bakchain=rt.TChain("event")
  trainchain=[rt.TChain("event"),rt.TChain("event")]
  for i in range(20):
    #sigchain.Add("/pad/yulee/geant4/tester/analysis/fast/uJet50GeV_fastsim_{}.root".format(i))
    #bakchain.Add("/pad/yulee/geant4/tester/analysis/fast/gJet50GeV_fastsim_{}.root".format(i))
    trainchain[0].Add("/pad/yulee/geant4/tester/analysis/fast/uJet50GeV_fastsim_{}.root".format(i))
    trainchain[1].Add("/pad/yulee/geant4/tester/analysis/fast/gJet50GeV_fastsim_{}.root".format(i))
  a=DataGenerator(trainchain,data_form="point",batch_size=512)
  b=a.__getitem__(10)
  b=a.__getitem__(10)
  b=a.__getitem__(10)
  path=["/pad/yulee/geant4/tester/analysis/fast/uJet50GeV_fastsim_{}.root","/pad/yulee/geant4/tester/analysis/fast/gJet50GeV_fastsim_{}.root"]
  train,val,test=prepare_data(path,data_form="point")
  print(train.total_len,val.total_len,test.total_len)
  x,y=test.get_test()
  print(x.shape,y.shape)
