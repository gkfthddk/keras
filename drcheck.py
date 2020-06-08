names=["qg023v50","qg123v50","qg223v50","qg124v50","qg224v50","qg50","qgrot50","qgrot1650","qg124rot50","qg124rot350","qg124rotb50","qg124rot250","qg124rot1650","qg124rot16250","qg124rot16350","qg124rot16450","qg124rot16550","qgor501024","qgor502048","qgor504096","qgor505120","qgnor505120"]
book={}
for na in names:
  try:
    f=open(na+".auc","r")
    li=f.readlines()
    f.close()
    drsum=0.
    if(";" in li[0]):
      dr2sum=[0.,0.,0.]
      for i in li:
        dr2split=i.split(";")
        for j in range(3):
          dr2sum[j]=dr2sum[j]+float(dr2split[j])
      dr2mean=[i/len(li) for i in dr2sum]
      print(na,dr2mean)
    else:
      for i in li:
        drsum+=float(i)
      book[na]=drsum/len(li)
      print(na,book[na])
  except:
    pass
