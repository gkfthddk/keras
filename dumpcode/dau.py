import ROOT as rt
events=["qq","zq","gg","zg","jj","zj"]
pts=[100,200,500,1000]
for l in events:
  for pt in pts:
    title=l+"_pt_"+str(pt)+"_"+str(int(pt*1.1))+".root"
    tfile=rt.TFile("Data/"+title,'read')
    jet=tfile.Get("jetAnalyser")
    print(l,pt,jet.GetEntriesFast())
