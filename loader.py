import numpy as np
from datetime import datetime
a=np.load("Data/jj_pt_1000_1100.npz")
qjetset=[]
print(len(a["pt"]))
now=datetime.now()
c=filter(lambda i : a["eta"][i]<0, range(1000))
#c=[i for i in range(10000) if a["eta"][i]<0]
#d=a["dau_pt"][c]
#[[a["dau_pt"][i],a["dau_deta"][i],a["dau_dphi"][i],a["dau_charge"][i]] for i in range(len(a["pt"])) if a["eta"][i]<0]
print(datetime.now()-now)
