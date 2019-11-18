import ROOT as rt
# python -i vardraw.py --vars major_axis,chad_mult --jets mg5_pp_gg_nocr,mg5_pp_zg_nocr,mg5_pp_gg_default,mg5_pp_zg_default,mg5_pp_gg_nocharm,mg5_pp_gg_pt_1000_1100 --cut etapt
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--jets",type=str,default="",help="")
parser.add_argument("--vars",type=str,default="pt",help="")
parser.add_argument("--cut",type=str,default="nocut",help="")
parser.add_argument("--frac",type=int,default=0,help="")
parser.add_argument("--binh",type=int,default=1,help="")
args=parser.parse_args()
jets=[]
jetf=[]
jetnames=args.jets.split(",")
for i in range(len(jetnames)):
  if("/" in jetnames[i]):jetf.append(rt.TFile("{}.root".format(jetnames[i]))) 
  else:jetf.append(rt.TFile("jets/{}.root".format(jetnames[i]))) 
  jets.append(jetf[i].Get("jetAnalyser"))
varss=args.vars.split(",")
canv=rt.TCanvas("canv","canv",1500,1500)
pt=1000
if(len(varss)!=1):
  canv.Divide(2,len(varss)/2+len(varss)%2)
cut=args.cut
hists=[]
for i in range(len(varss)):
  cutter=""
  canv.cd(i+1)
  ma=0
  mi=0
  mv=0
  for j in range(len(jets)):
    if(ma<jets[j].GetMaximum(varss[i])):ma=jets[j].GetMaximum(varss[i])
    if(mi>jets[j].GetMinimum(varss[i])):mi=jets[j].GetMinimum(varss[i])
  binh=args.binh
  if("chad" in varss[i]):
    hists.append(rt.TH1F("hist{}".format(varss[i]),"{}-{}".format(pt,varss[i]),int(ma/binh),0,int(ma)))
  elif("nhad" in varss[i]):
    hists.append(rt.TH1F("hist{}".format(varss[i]),"{}-{}".format(pt,varss[i]),int(ma),0,int(ma)))
  elif("eta" in varss[i]):
    hists.append(rt.TH1F("hist{}".format(varss[i]),"{}-{}".format(pt,varss[i]),int(100/binh),-2.4,2.4))
  else:
    hists.append(rt.TH1F("hist{}".format(varss[i]),"{}-{}".format(pt,varss[i]),int(100/binh),mi,ma))
  for j in range(len(jets)):
    res=60
    if("chad" in varss[i]):
      hists.append(rt.TH1F("{}hist{}".format(j,varss[i]),"{}".format(jetnames[j]),int(ma/binh),0,int(ma)))
    elif("nhad" in varss[i]):
      hists.append(rt.TH1F("{}hist{}".format(j,varss[i]),"{}".format(jetnames[j]),int(ma),0,int(ma)))
    elif("eta" in varss[i]):
      hists.append(rt.TH1F("{}hist{}".format(j,varss[i]),"{}".format(jetnames[j]),int(100/binh),-2.4,2.4))
    else:
      hists.append(rt.TH1F("{}hist{}".format(j,varss[i]),"{}".format(jetnames[j]),int(100/binh),mi,ma))
    if("eta" == cut):
      cutter="eta<1.&&eta>-1."
    if("pt" == cut):
      cutter="pt<{}&&pt>{}".format(1.076*pt,0.8235*pt) 
    if("etapt" == cut):
      cutter="eta<1.&&eta>-1.&&pt<{}&&pt>{}".format(1.076*pt,0.8235*pt) 
    jets[j].Draw("{}>>{}hist{}".format(varss[i],j,varss[i]),cutter)

    #for j in [1,3]:
    hists[i*(len(jets)+1)+j+1].Scale(1.0/hists[i*(len(jets)+1)+j+1].Integral())
    if(mv<hists[i*(len(jets)+1)+j+1].GetMaximum()):mv=hists[i*(len(jets)+1)+j+1].GetMaximum()
    hists[i*(len(jets)+1)+j+1].SetLineColor(j+1)
    #mv=3
  hists[i*(len(jets)+1)].SetMaximum(mv*1.3)
  if(args.frac==1):
    hists[i*(len(jets)+1)].SetTitle("zg/gg-{}-{}".format(pt,varss[i]))
    hists[i*(len(jets)+1)].SetMaximum(2)
    
  hists[i*(len(jets)+1)].Draw()
  for j in range(len(jets)):
    if(args.frac==1):
      if(j%2==0):
        hists[i*(len(jets)+1)+j+1].Divide(hists[i*(len(jets)+1)+j+2])
        hists[i*(len(jets)+1)+j+1].SetLineColor(j/2+1)
      if(j%2==1):continue
    hists[i*(len(jets)+1)+j+1].Draw("Same")
  rt.gStyle.SetOptStat(False)
  rt.gPad.BuildLegend()
    #canv.SaveAs("bdtvars/newnocr{}_{}.png".format(pt,cut))
  #ggdata.Close()
