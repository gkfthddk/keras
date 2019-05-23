import os

saves=["zjcnn","zjrnn","bdt"]
#saves=["zqcnn","qqcnn"]
pts=[100,200,500,1000]
conds=["","pt","noeta"]
#gets=["nocut","eta","acut"]
#saves=["zjcnn"]
#pts=[1000]
#conds=[""]
gets=["nocut","eta"]

for save in saves:
  try:
    for pt in pts:
      for cond in conds:
        for get in gets:
          if(cond=="noeta"):os.system("python getroc.py --save asu{save}{pt}{cond} --pt {pt} --get {get}".format(save=save,cond=cond,pt=pt,get=get))
          else:os.system("python getroc.py --save asu{save}{pt}{cond} --pt {pt} --get {get} &".format(save=save,cond=cond,pt=pt,get=get))
  except:print("error")
