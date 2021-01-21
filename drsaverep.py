import numpy as np
import ROOT as rt
import datetime
now=datetime.datetime.now()
_fb=56*3
#_cb=8
_cb=0
num_point=2048
event=rt.TChain("event")
event.Add("~/dream/elenshower.root")
#event.Add("~/geant4/tester/analysis/el20.root")

images=[]
points=[]
el_pt_gen=[]
el_dr_ecorr=[]
print(event.GetEntries())
for i in range(int(event.GetEntries())):
    event.GetEntry(i)
    el_pt_gen.append(event.pt_Gen/1000.)
    el_dr_ecorr.append(event.E_DRcorr)
    images.append([np.array(list(event.image_ecor_s)[:_fb*_fb]).reshape((_fb,_fb))[_cb:_fb-_cb,_cb:_fb-_cb],
        np.array(list(event.image_ecor_c)[:_fb*_fb]).reshape((_fb,_fb))[_cb:_fb-_cb,_cb:_fb-_cb],
        np.array(list(event.image_n_s)[:_fb*_fb]).reshape((_fb,_fb))[_cb:_fb-_cb,_cb:_fb-_cb],
        np.array(list(event.image_n_s)[:_fb*_fb]).reshape((_fb,_fb))[_cb:_fb-_cb,_cb:_fb-_cb]])
    point=[np.array(list(event.fiber_phi)),
        np.array(list(event.fiber_eta)),
        np.array(list(event.fiber_depth)),
        np.array(list(event.fiber_ecor_s)),
        np.array(list(event.fiber_ecor_c)),
        ]
    point=sorted(point, key=lambda pnt:pnt[3],reverse=True)
    if(len(point[0])<2048):
        point=np.concatenate([point,np.zeros((5,2048-len(point[0])))],axis=1)
    if(len(point[0])>2048):
        point=np.array(point)[:,:2048]
    points.append(point)

print("depth",np.mean(list(event.fiber_depth)))
el_image=np.array(images)
print([len(i) for i in point])
el_point=np.array(points)
print(el_image.shape,el_point.shape)
del event,images,points
event=rt.TChain("event")
event.Add("~/dream/pienshower.root")
#event.Add("~/geant4/tester/analysis/pi20.root")

images=[]
points=[]
pi_pt_gen=[]
pi_dr_ecorr=[]
for i in range(int(event.GetEntries())):
    event.GetEntry(i)
    pi_pt_gen.append(event.pt_Gen/1000.)
    pi_dr_ecorr.append(event.E_DRcorr)
    images.append([np.array(list(event.image_ecor_s)[:_fb*_fb]).reshape((_fb,_fb))[_cb:_fb-_cb,_cb:_fb-_cb],
        np.array(list(event.image_ecor_c)[:_fb*_fb]).reshape((_fb,_fb))[_cb:_fb-_cb,_cb:_fb-_cb],
        np.array(list(event.image_n_s)[:_fb*_fb]).reshape((_fb,_fb))[_cb:_fb-_cb,_cb:_fb-_cb],
        np.array(list(event.image_n_s)[:_fb*_fb]).reshape((_fb,_fb))[_cb:_fb-_cb,_cb:_fb-_cb]])
    point=[np.array(list(event.fiber_phi)),
        np.array(list(event.fiber_eta)),
        np.array(list(event.fiber_depth)),
        np.array(list(event.fiber_ecor_s)),
        np.array(list(event.fiber_ecor_c)),
        ]
    point=sorted(point, key=lambda pnt:pnt[3],reverse=True)
    if(len(point[0])<2048):
        point=np.concatenate([point,np.zeros((5,2048-len(point[0])))],axis=1)
    if(len(point[0])>2048):
        point=np.array(point)[:,:2048]
    points.append(point)
pi_image=np.array(images)
pi_point=np.array(points)
print(pi_image.shape,pi_point.shape)
del event,images,points
#np.savez("npzs/dregp20pimg",el=el,ga=ga,pi=pi,pi0=pi0)
np.savez("npzs/drep",el_image=el_image,el_point=el_point,el_pt_gen=el_pt_gen,el_dr_ecorr=el_dr_ecorr,pi_image=pi_image,pi_point=pi_point,pi_pt_gen=pi_pt_gen,pi_dr_ecorr=pi_dr_ecorr)
print("time costs ",datetime.datetime.now()-now)
