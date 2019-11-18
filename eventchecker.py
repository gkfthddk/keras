import ROOT as rt

f=rt.TFile("Data/jj_pt_100_110.root",'read')
jet=f.Get("jetAnalyser")
a=-1
b=0
for i in range(jet.GetEntries()):
  jet.GetEntry(i)
  if(a==jet.event):
    b+=1
  else:
    if(b==1):print(i,b)
    a=jet.event
    b=1
  if(b>2):print(i,b)
