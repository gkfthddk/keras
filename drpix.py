import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

a=np.load("side2350img.npz")
#a=np.load("rot50img.npz")
b=a["imgset"].item()
c=b["gj"]
d=np.zeros(c[0][0].shape)
d=c[474][0]
#for i in range(len(c)):
#  d+=c[i][4]

plt.imshow(d,origin='lower')
"""
vox=a["voxels"].item()
gj=vox["gj"]
qj=vox["uj"]
def explode(data):
  size = np.array(data.shape)*2
  data_e = np.zeros(size - 1, dtype=data.dtype)
  data_e[::2, ::2, ::2] = data
  return data_e
def tohex(val):
  alpha=format(val,"x")
  if(len(alpha)==1):
    alpha="0"+alpha
  return alpha
iyzx_e_s=np.zeros((gj[0][0].shape))
for i in range(len(gj)):
  iyzx_e_s=iyzx_e_s+gj[i][0]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xyz = np.zeros(iyzx_e_s.shape,dtype='S9')
iyzxgj=iyzx_e_s
iyzx_e_s=255.*iyzx_e_s/iyzx_e_s.max()
for i in range(len(xyz)):
  for j in range(len(xyz)):
    for k in range(len(xyz)):
      #xyz[i,j,k]="#0000"+tohex(int(255*pow(1.2,-(pow(i-3,2)+pow(j-3,2)+pow(k-3,2)))))+tohex(int(20*pow(1.2,-(pow(i-3,2)+pow(j-3,2)+pow(k-3,2)))))
      xyz[i,j,k]="#0000ff"+tohex(int(iyzx_e_s[i,j,k]))
      #xyz[i,j,k]="#"+tohex(255-int(iyzx_e_s[i,j,k]))*2+"ff"+tohex(int(iyzx_e_s[i,j,k]))

alpha=tohex(20)
facecolors=xyz
#facecolors = np.where(n_voxels, '#FFD65D'+alpha, '#7A88CC'+alpha)
filled = np.ones(iyzx_e_s.shape)
edgecolors = np.where(filled, '#000000'+"00", '#000000'+"00")

# upscale the above voxel image, leaving gaps
filled_2 = explode(filled)
fcolors_2 = explode(facecolors)
ecolors_2 = explode(edgecolors)
x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
x[0::2, :, :] += 0.05
y[:, 0::2, :] += 0.05
z[:, :, 0::2] += 0.05
x[1::2, :, :] += 0.95
y[:, 1::2, :] += 0.95
z[:, :, 1::2] += 0.95

ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, edgecolors=ecolors_2)
ax.set_title("Gluon")
ax.set_xlabel("phi")
ax.set_ylabel("eta")
ax.set_zlabel("depth")

iyzx_e_s=np.zeros((qj[0][0].shape))
print(iyzx_e_s.shape)
for i in range(len(qj)):
  iyzx_e_s=iyzx_e_s+qj[i][0]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xyz = np.zeros(iyzx_e_s.shape,dtype='S9')
iyzxqj=iyzx_e_s
iyzx_e_s=255.*iyzx_e_s/iyzx_e_s.max()
for i in range(len(xyz)):
  for j in range(len(xyz)):
    for k in range(len(xyz)):
      #xyz[i,j,k]="#0000"+tohex(int(255*pow(1.2,-(pow(i-3,2)+pow(j-3,2)+pow(k-3,2)))))+tohex(int(20*pow(1.2,-(pow(i-3,2)+pow(j-3,2)+pow(k-3,2)))))
      xyz[i,j,k]="#0000ff"+tohex(int(iyzx_e_s[i,j,k]))
      #xyz[i,j,k]="#"+tohex(255-int(iyzx_e_s[i,j,k]))*2+"ff"+tohex(int(iyzx_e_s[i,j,k]))

alpha=tohex(20)
facecolors=xyz
#facecolors = np.where(n_voxels, '#FFD65D'+alpha, '#7A88CC'+alpha)
filled = np.ones(iyzx_e_s.shape)
edgecolors = np.where(filled, '#000000'+"00", '#000000'+"00")

# upscale the above voxel image, leaving gaps
filled_2 = explode(filled)
fcolors_2 = explode(facecolors)
ecolors_2 = explode(edgecolors)
x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
x[0::2, :, :] += 0.05
y[:, 0::2, :] += 0.05
z[:, :, 0::2] += 0.05
x[1::2, :, :] += 0.95
y[:, 1::2, :] += 0.95
z[:, :, 1::2] += 0.95

ax.voxels(x, y, z, filled_2, facecolors=fcolors_2, edgecolors=ecolors_2)
ax.set_title("Quark")
ax.set_xlabel("phi")
ax.set_ylabel("eta")
ax.set_zlabel("depth")
"""
plt.show()
