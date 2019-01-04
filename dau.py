from iter import *

dau=wkiter(["../jetdata/quark100_img.root","../jetdata/gluon100_img.root"])
#dau=wkiter(["/scratch/yjdata/ppzj100__img.root","/scratch/yjdata/ppjj100__img.root"])
gen=dau.next()
a,b=next(gen)
gjet=dau.gjet
qjet=dau.qjet
