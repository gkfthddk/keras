import autokeras as ak
import os
import numpy as np
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(4)

loaded=np.load("jet{}.npz".format(500))
X=loaded["seqset"]
Y=loaded["labelset"]
pidset=loaded["pidset"]
xb=[]
#X=np.reshape(X,(-1,1,X.shape[-2],X.shape[-1]))
#for x in X:
#  xb.append([x[1]])
#X=np.array(xb)

label1=[]
for i in range(len(Y)):
  xb.extend([X[i][0],X[i][1]])
  if(Y[i][0]==1):
    label1.append([1,0])
    label1.append([1,0])
  elif(Y[i][1]==1):
    label1.append([1,0])
    label1.append([0,1])
  elif(Y[i][2]==1):
    label1.append([0,1])
    label1.append([1,0])
  elif(Y[i][3]==1):
    label1.append([0,1])
    label1.append([0,1])
label=np.array(label1)[:,0]
xb=np.array(xb)
clf = ak.ImageClassifier(path='autokeras/',verbose=True)
clf.fit(xb[:5000], label[:5000],time_limit=60*60)
#results = clf.predict(X)
clf.evaluate(np.array(xb),label1[:,0])
