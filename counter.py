import matplotlib.pyplot as plt
li=[('pepqqcnn100model', 'v3t3', 0.4999833416625021, 0.937),
('pepqqcnn200model', 'v3t3', 0.5000670915800067, 0.943),
('pepqqcnn500model', 'v3t3', 0.49996678844237796, 0.949),
('pepqqcnn1000model', 'v3t3', 0.500033380065425, 0.957),
('pepzjcnn100model', 'v1t2', 0.4999328092454478, 0.926),
('pepzjcnn200model', 'v1t2', 0.5000502765208648, 0.936),
('pepzjcnn500model', 'v1t2', 0.5000671095899604, 0.942),
('pepzjcnn1000model', 'v1t2', 0.4999001929602768, 0.921),
('pepzjcnn100model', 'v1t3', 0.4999665954035275, 0.925),
('pepzjcnn200model', 'v1t3', 0.5000336428475306, 0.917),
('pepzjcnn500model', 'v1t3', 0.5000833639000967, 0.918),
('pepzjcnn1000model', 'v1t3', 0.4999500582653571, 0.901),
('pepzqcnn100model', 'v2t2', 0.5000501119166137, 0.941),
('pepzqcnn200model', 'v2t2', 0.500033505327347, 0.954),
('pepzqcnn500model', 'v2t2', 0.5000167453699053, 0.964),
('pepzqcnn1000model', 'v2t2', 0.4999167360532889, 0.968)]
plt.figure(figsize=(12,9))
plt.grid(True)
pl=[]
for i in range(4):
  pl.append([[],[]])
  pl[i][0].append(100)
  pl[i][0].append(200)
  pl[i][0].append(500)
  pl[i][0].append(1000)
  pl[i][1].append(li[i*4+0][3])
  pl[i][1].append(li[i*4+1][3])
  pl[i][1].append(li[i*4+2][3])
  pl[i][1].append(li[i*4+3][3])
  name=""
  a=li[i*4][0]
  b=li[i*4][1]
  if("zj" in a):name+="generic"
  if("zq" in a):name+="Z+jet"
  if("qq" in a):name+="dijet"
  if("t2" in b):name+="-Z+jet"
  if("t3" in b):name+="-dijet"
  if("cnn" in a):name+="-CNN"
  if("rnn" in a):name+="-RNN"
  pl[i].append(name)
  plt.plot(pl[i][0],pl[i][1],label=name)
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.,fontsize=22)
plt.savefig("test",bbox_inches='tight',dpi=100)
