import ROOT as rt
#python -i deldraw.py --vars nhad_mult,chad_mult --jets mg5_pp_gg_default_1,mg5_pp_gg_nocr_1,mg5_pp_gg_nocharm_1,mg5_pp_zg_default_1,mg5_pp_zg_nocr_1,mg5_pp_gg_pt_1000_1100_1,mg5_pp_zg_pt_1000_1100_1,mg5_pp_gg_pt_1000_1100_30 --cut etapt --binh 2
#python -i deldraw.py --vars pt,eta,nhad_mult,chad_mult --jets mg5_pp_gg_default_1,mg5_pp_gg_pt_1000_1100_1,mg5_pp_gg_nocr_1,mg5_pp_gg_nocharm_1,mg5_pp_zg_pt_1000_1100_1 --cut nocut
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
  print(jetnames)
  if("/" in jetnames[i]):
    jetf.append(rt.TFile("{}.root".format(jetnames[i]))) 
    jets.append(jetf[i].Get("jetAnalyser"))
  else:
    jetf.append(rt.TFile("jets/{}.root".format(jetnames[i]))) 
    jets.append(jetf[i].Get("Delphes"))
varss=[]
for var in args.vars.split(","):
  if("chad" in var):varss.append("Jet.NCharged")
  if("nhad" in var):varss.append("Jet.NNeutrals")
  if("pt" == var):varss.append("Jet.PT")
  if("ptd" == var):varss.append("Jet.PTD")
  if("eta" == var):varss.append("Jet.Eta")

canv=rt.TCanvas("canv","canv",1200,1200)
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
  if("Charged" in varss[i]):
    hists.append(rt.TH1F("hist{}".format(varss[i]),"{}-{}".format(pt,varss[i]),int((ma-10)/binh),10,int(ma)))
  elif("Neutrals" in varss[i]):
    hists.append(rt.TH1F("hist{}".format(varss[i]),"{}-{}".format(pt,varss[i]),int(ma),0,int(ma)))
  elif("Eta" in varss[i]):
    hists.append(rt.TH1F("hist{}".format(varss[i]),"{}-{}".format(pt,varss[i]),int(100/binh),-2.4,2.4))
  else:
    hists.append(rt.TH1F("hist{}".format(varss[i]),"{}-{}".format(pt,varss[i]),int(100/binh),mi,ma))
  for j in range(len(jets)):
    res=60
    if("Charged" in varss[i]):
      hists.append(rt.TH1F("{}hist{}".format(j,varss[i]),"{}".format(jetnames[j]),int((ma-10)/binh),10,int(ma)))
    elif("Neutrals" in varss[i]):
      hists.append(rt.TH1F("{}hist{}".format(j,varss[i]),"{}".format(jetnames[j]),int(ma),0,int(ma)))
    elif("Eta" in varss[i]):
      hists.append(rt.TH1F("{}hist{}".format(j,varss[i]),"{}".format(jetnames[j]),int(100/binh),-2.4,2.4))
    else:
      hists.append(rt.TH1F("{}hist{}".format(j,varss[i]),"{}".format(jetnames[j]),int(100/binh),mi,ma))
    if("eta" == cut):
      cutter="Jet.Eta<1.&&Jet.Eta>-1."
    if("pt" == cut):
      cutter="Jet.PT<{}&&Jet.PT>{}".format(1.076*pt,0.8235*pt) 
    if("etapt" == cut):
      cutter="Jet.Eta<1.&&Jet.Eta>-1.&&Jet.PT<{}&&Jet.PT>{}".format(1.076*pt,0.8235*pt) 
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
