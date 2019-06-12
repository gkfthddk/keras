import os

#saves=["zjcnn",'bdt',"zjrnn"]
#saves=["zjcnn","zjrnn","bdt"]
saves=["zqcnn","qqcnn"]
pts=[100,200,500,1000]
#pts=[100]
#conds=["ptonly","pteta","ptonly2","pteta31","pteta21"]
conds=["pteta"]
#conds=["ptonlyadam","ptetaadam"]
#gets=["nocut","eta","acut"]
#saves=["zjcnn"]
#pts=[1000]
gets=["acut"]
#gets=["nocut","pteta"]

for save in saves:
  try:
    for pt in pts:
      for cond in conds:
        for get in gets:
          if(cond=="noeta"):os.system("python getauc.py --save asu{save}{pt}{cond} --pt {pt} --get {get}".format(save=save,cond=cond,pt=pt,get=get))
          else:os.system("python getauc.py --save asu{save}{pt}{cond} --pt {pt} --get {get} &".format(save=save,cond=cond,pt=pt,get=get))
  except:print("error")
