import os
pts=[100,200,500,1000]
for etabin in [1.,2.4]:
  for onpt in [0,1]:
    if(etabin==2.4):
      ptroc="python getroc.py --save asuzjcnn100pt --pt 100 --etabin {etabin}  &,python getroc.py --save asuzjcnn200pt --pt 200 --etabin {etabin} &, python getroc.py --save asuzjcnn500pt --pt 500 --etabin {etabin} &,python getroc.py --save asuzjcnn1000pt --pt 1000 --etabin {etabin} &, python getroc.py --save asubdt100pt --pt 100 --etabin {etabin}  &, python getroc.py --save asubdt200pt --pt 200 --etabin {etabin}  &, python getroc.py --save asubdt500pt --pt 500 --etabin {etabin}  &, python getroc.py --save asubdt1000pt --pt 1000 --etabin {etabin}  &,".format(etabin=etabin)
      if(onpt):ptroc="python getroc.py --save asuzjcnn100pt --pt 100 --ptmin 0.815 --ptmax 1.159 --get {get} --etabin {etabin}  &,python getroc.py --save asuzjcnn200pt --pt 200 --ptmin 0.819 --ptmax 1.123 --get {get} --etabin {etabin} &, python getroc.py --save asuzjcnn500pt --pt 500 --ptmin 0.821 --ptmax 1.093 --get {get} --etabin {etabin} &,python getroc.py --save asuzjcnn1000pt --pt 1000 --ptmin 0.8235 --ptmax 1.076 --get {get} --etabin {etabin} &, python getroc.py --save asubdt100pt --pt 100 --ptmin 0.815 --ptmax 1.159 --get {get} --etabin {etabin}  &, python getroc.py --save asubdt200pt --pt 200 --ptmin 0.819 --ptmax 1.123 --get {get} --etabin {etabin}  &, python getroc.py --save asubdt500pt --pt 500 --ptmin 0.821 --ptmax 1.093 --get {get} --etabin {etabin}  &, python getroc.py --save asubdt1000pt --pt 1000 --ptmin 0.8235 --ptmax 1.076 --get {get} --etabin {etabin}  &,".format(get='pt',etabin=etabin)
    else:
      ptroc="python getroc.py --save asuzjcnn100pt --pt 100 --get {get} --etabin {etabin}  &,python getroc.py --save asuzjcnn200pt --pt 200 --get {get} --etabin {etabin} &, python getroc.py --save asuzjcnn500pt --pt 500 --get {get} --etabin {etabin} &,python getroc.py --save asuzjcnn1000pt --pt 1000 --get {get} --etabin {etabin} &, python getroc.py --save asubdt100pt --pt 100 --get {get} --etabin {etabin}  &, python getroc.py --save asubdt200pt --pt 200 --get {get} --etabin {etabin}  &, python getroc.py --save asubdt500pt --pt 500 --get {get} --etabin {etabin}  &, python getroc.py --save asubdt1000pt --pt 1000 --get {get} --etabin {etabin}  &,".format(get='eta',etabin=etabin)
      if(onpt):ptroc="python getroc.py --save asuzjcnn100pt --pt 100 --ptmin 0.815 --ptmax 1.159 --get {get} --etabin {etabin}  &, python getroc.py --save asuzjcnn200pt --pt 200 --ptmin 0.819 --ptmax 1.123 --get {get} --etabin {etabin} &, python getroc.py --save asuzjcnn500pt --pt 500 --ptmin 0.821 --ptmax 1.093 --get {get} --etabin {etabin} &, python getroc.py --save asuzjcnn1000pt --pt 1000 --ptmin 0.8235 --ptmax 1.076 --get {get} --etabin {etabin} &, python getroc.py --save asubdt100pt --pt 100 --ptmin 0.815 --ptmax 1.159 --get {get} --etabin {etabin}  &, python getroc.py --save asubdt200pt --pt 200 --ptmin 0.819 --ptmax 1.123 --get {get} --etabin {etabin}  &, python getroc.py --save asubdt500pt --pt 500 --ptmin 0.821 --ptmax 1.093 --get {get} --etabin {etabin}  &, python getroc.py --save asubdt1000pt --pt 1000 --ptmin 0.8235 --ptmax 1.076 --get {get} --etabin {etabin}  &,".format(get='pteta',etabin=etabin)

    gausroc=""
    for pt in pts:
      if(etabin==2.4):
        if(onpt):gausroc+="python getroc.py --save asuzjcnn{pt}ptgaus --pt {pt} --get {get} --etabin {etabin} --gaus 1 ,".format(pt=pt,get='gaus',etabin=etabin)
        else:gausroc+="python getroc.py --save asuzjcnn{pt}ptgaus --pt {pt} --etabin {etabin} ,".format(pt=pt,etabin=etabin)
      else:
        if(onpt):gausroc+="python getroc.py --save asuzjcnn{pt}ptgaus --pt {pt} --get {get} --etabin {etabin} --gaus 1 ,".format(pt=pt,get='gauseta',etabin=etabin)
        else:gausroc+="python getroc.py --save asuzjcnn{pt}ptgaus --pt {pt} --get {get} --etabin {etabin} ,".format(pt=pt,get='eta',etabin=etabin)

    etaroc=""
    for pt in pts:
      if(etabin==2.4):
        etaroc+="python getroc.py --save asuzjcnn{pt} --pt {pt} --etabin {etabin} &, python getroc.py --save asuzjcnn{pt}noeta --pt {pt} --etabin {etabin} &, python getroc.py --save asubdt{pt} --pt {pt} --etabin {etabin} &, python getroc.py --save asubdt{pt}noeta --pt {pt} --etabin {etabin} &,".format(pt=pt,etabin=etabin)
      else:
        etaroc+="python getroc.py --save asuzjcnn{pt} --pt {pt} --get {get} --etabin {etabin} &, python getroc.py --save asuzjcnn{pt}noeta --pt {pt} --get {get} --etabin {etabin} &, python getroc.py --save asubdt{pt} --pt {pt} --get {get} --etabin {etabin} &, python getroc.py --save asubdt{pt}noeta --pt {pt} --get {get} --etabin {etabin} &,".format(pt=pt,get='eta',etabin=etabin)

    ptroc=ptroc.split(',')
    gausroc=gausroc.split(',')
    etaroc=etaroc.split(',')
    for i in ptroc:
      os.system(i)
    for i in gausroc:
      os.system(i)
    for i in etaroc:
      os.system(i)
    os.system('echo ""')

"""
python getroc.py --save asuzjcnn100pt --pt 100 --ptmin 0.815 --ptmax 1.159 --get pt --etabin {etabin}  &,
python getroc.py --save asuzjcnn200pt --pt 200 --ptmin 0.819 --ptmax 1.123 --get pt --etabin {etabin} &,
python getroc.py --save asuzjcnn500pt --pt 500 --ptmin 0.821 --ptmax 1.093 --get pt --etabin {etabin} &,
python getroc.py --save asuzjcnn1000pt --pt 1000 --ptmin 0.8235 --ptmax 1.076 --get pt --etabin {etabin} &,
python getroc.py --save asuzjcnn100 --pt 100 --etabin {etabin} &,
python getroc.py --save asuzjcnn200 --pt 200 --etabin {etabin} &,
python getroc.py --save asuzjcnn500 --pt 500 --etabin {etabin} &,
python getroc.py --save asuzjcnn1000 --pt 1000 --etabin {etabin} &,
python getroc.py --save asuzjcnn100noeta --pt 100 --etabin {etabin} &,
python getroc.py --save asuzjcnn200noeta --pt 200 --etabin {etabin} &,
python getroc.py --save asuzjcnn500noeta --pt 500 --etabin {etabin} &,
python getroc.py --save asuzjcnn1000noeta --pt 1000 --etabin {etabin} &,
python getroc.py --save asuzjcnn100ptgaus --pt 100 --get gaus --etabin 1  --gaus {etabin} &,
python getroc.py --save asuzjcnn200ptgaus --pt 200 --get gaus --etabin 1  --gaus {etabin} &,
python getroc.py --save asuzjcnn500ptgaus --pt 500 --get gaus --etabin 1  --gaus {etabin} &,
python getroc.py --save asuzjcnn1000ptgaus --pt 1000 --get gauseta --etabin 1  --gaus {etabin} &,
python getroc.py --save asubdt100pt --pt 100 --ptmin 0.815 --ptmax 1.159 --get pt --etabin {etabin}  &,
python getroc.py --save asubdt200pt --pt 200 --ptmin 0.819 --ptmax 1.123 --get pt --etabin {etabin}  &,
python getroc.py --save asubdt500pt --pt 500 --ptmin 0.821 --ptmax 1.093 --get pt --etabin {etabin}  &,
python getroc.py --save asubdt1000pt --pt 1000 --ptmin 0.8235 --ptmax 1.076 --get pt --etabin {etabin}  &,
python getroc.py --save asubdt100 --pt 100 --etabin {etabin} &,
python getroc.py --save asubdt200 --pt 200 --etabin {etabin} &,
python getroc.py --save asubdt500 --pt 500 --etabin {etabin} &,
python getroc.py --save asubdt1000 --pt 1000 --etabin {etabin} &,
python getroc.py --save asubdt100noeta --pt 100 --etabin {etabin} &,
python getroc.py --save asubdt200noeta --pt 200 --etabin {etabin} &,
python getroc.py --save asubdt500noeta --pt 500 --etabin {etabin} &,
python getroc.py --save asubdt1000noeta --pt 1000 --etabin {etabin}
"""
