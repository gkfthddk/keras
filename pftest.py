import numpy as np
import matplotlib.pyplot as plt
import ROOT as rt
f=np.load("pf200.npz")
seq=f["seqset"]
imgset=f["imgset"]
eta=f["etaset"]
phi=f["phiset"]
cyl=rt.TH2F("hist","hist",72,-rt.TMath.Pi(),rt.TMath.Pi(),55,-2.4,2.4)
jet=rt.TH2F("his2t","his2t",72,-rt.TMath.Pi(),rt.TMath.Pi(),55,-2.4,2.4)
rt.gStyle.SetOptStat(0)
ent=len(seq)
img=np.zeros(55*72)
#img=np.zeros((55,72))
phibin,etabin=[0,0]
def dr(i=None):
  if(i==None):i=np.random.randint(ent)
  print(i)
  cyl.Reset()
  jet.Reset()
  img=np.zeros(55*72)
  for j in range(len(seq[i])):
    if(abs(seq[i][j][1])>2.4):continue
    if(abs(seq[i][j][2])>3.14159):continue
    if(j<2):
      jet.Fill(phi[i][j],eta[i][j],100*(2-j))
    if(seq[i][j][0]==0):
      break
    #theta=2.*rt.TMath.ATan(rt.TMath.Exp(-seq[i][j][1]))
    #phi=seq[i][j][2]
    cyl.Fill(seq[i][j][2],seq[i][j][1],1)
    #cyl.Fill(seq[i][j][2],seq[i][j][1],seq[i][j][0])
    etabin=int(55*(seq[i][j][1]+2.4)/(2*2.4))
    phibin=int(72*(seq[i][j][2]+3.14159)/(2*3.14159))
    pix=72*int(etabin)+int(phibin)
    img[pix]=img[pix]+1.
    #img[etabin,phibin]=seq[i][j][0]
  #print("mean dau",d/100.,d)
  #import matplotlib.pyplot as plt
  #plt.hist(d)
  #plt.show()
  #cyl.Draw("surf5 cyl")
  cyl.Draw("colz")
  #jet.Draw("surf cyl same")
  jet.Draw("box same")
  plt.imshow((imgset[i][2]+imgset[i][3]).reshape((55,72)),origin='lower',cmap="viridis")
  #plt.imshow(img.reshape((55,72))/img.max(),origin='lower',cmap="viridis")
  plt.show()
dr(np.random.randint(ent))
