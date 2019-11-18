import numpy as np

for pt in [100,200,500,1000]:
  a=np.load("jet{}.npz".format(pt))
  label=a["labelset"]
  show=np.unique([np.where(label[i]==1)[0][0] for i in range(len(label))],return_counts=1)
  print(pt,[round(1.*show[1][i]/sum(show[1]),3) for i in range(4)])
