import ROOT as rt
for pt in [100,200,500,1000]:
  for cut in ["nocut","eta","pt","etapt"]:
    zqdata=rt.TFile("Data/zq_pt_{}_{}.root".format(pt,int(pt*1.1)))
    zgdata=rt.TFile("Data/zg_pt_{}_{}.root".format(pt,int(pt*1.1)))
    qqdata=rt.TFile("Data/qq_pt_{}_{}.root".format(pt,int(pt*1.1)))
    ggdata=rt.TFile("Data/gg_pt_{}_{}.root".format(pt,int(pt*1.1)))
    zq=zqdata.Get("jetAnalyser")
    zg=zgdata.Get("jetAnalyser")
    qq=qqdata.Get("jetAnalyser")
    gg=ggdata.Get("jetAnalyser")
    varss=['pt','eta','ptd','major_axis','minor_axis','chad_mult','nhad_mult']
    #varss=['pt','eta','ptd']
    cavn=1
    cutter=""
    canv=rt.TCanvas("canv","canv",1000,1500)
    canv.Divide(2,4)
    hists=[]
    for i in range(len(varss)):
      canv.cd(cavn)
      cavn+=1
      ma=0;
      mi=0;
      if(ma<zq.GetMaximum(varss[i])):ma=zq.GetMaximum(varss[i])
      if(mi>zq.GetMinimum(varss[i])):ma=zq.GetMinimum(varss[i])
      if(ma<zg.GetMaximum(varss[i])):ma=zg.GetMaximum(varss[i])
      if(mi>zg.GetMinimum(varss[i])):ma=zg.GetMinimum(varss[i])
      if(ma<qq.GetMaximum(varss[i])):ma=qq.GetMaximum(varss[i])
      if(mi>qq.GetMinimum(varss[i])):ma=qq.GetMinimum(varss[i])
      if(ma<gg.GetMaximum(varss[i])):ma=gg.GetMaximum(varss[i])
      if(mi>gg.GetMinimum(varss[i])):ma=gg.GetMinimum(varss[i])
      res=60
      if("mult" in varss[i]):
        hists.append(rt.TH1F("hist{}".format(varss[i]),"{}-{}".format(pt,varss[i]),int(ma),0,int(ma)))
        hists.append(rt.TH1F("zqhist{}".format(varss[i]),"zq {}".format(varss[i]),int(ma),0,int(ma)))
        hists.append(rt.TH1F("zghist{}".format(varss[i]),"zg {}".format(varss[i]),int(ma),0,int(ma)))
        hists.append(rt.TH1F("qqhist{}".format(varss[i]),"qq {}".format(varss[i]),int(ma),0,int(ma)))
        hists.append(rt.TH1F("gghist{}".format(varss[i]),"gg {}".format(varss[i]),int(ma),0,int(ma)))
      elif("eta" in varss[i]):
        hists.append(rt.TH1F("hist{}".format(varss[i]),"{}-{}".format(pt,varss[i]),100,-2.4,2.4))
        hists.append(rt.TH1F("zqhist{}".format(varss[i]),"zq {}".format(varss[i]),100,-2.4,2.4))
        hists.append(rt.TH1F("zghist{}".format(varss[i]),"zg {}".format(varss[i]),100,-2.4,2.4))
        hists.append(rt.TH1F("qqhist{}".format(varss[i]),"qq {}".format(varss[i]),100,-2.4,2.4))
        hists.append(rt.TH1F("gghist{}".format(varss[i]),"gg {}".format(varss[i]),100,-2.4,2.4))
      else:
        hists.append(rt.TH1F("hist{}".format(varss[i]),"{}-{}".format(pt,varss[i]),100,mi,ma))
        hists.append(rt.TH1F("zqhist{}".format(varss[i]),"zq {}".format(varss[i]),100,mi,ma))
        hists.append(rt.TH1F("zghist{}".format(varss[i]),"zg {}".format(varss[i]),100,mi,ma))
        hists.append(rt.TH1F("qqhist{}".format(varss[i]),"qq {}".format(varss[i]),100,mi,ma))
        hists.append(rt.TH1F("gghist{}".format(varss[i]),"gg {}".format(varss[i]),100,mi,ma))
      #zq.Draw("{}>>zqhist".format(var),"pt<600.&&pt>400.")
      #zg.Draw("{}>>zghist".format(var),"pt<600.&&pt>400.")
      #qq.Draw("{}>>qqhist".format(var),"pt<600.&&pt>400.")
      #gg.Draw("{}>>gghist".format(var),"pt<600.&&pt>400.")
      if("eta" == cut):
        cutter="eta<1.&&eta>-1."
      if("pt" == cut):
        cutter="pt<{}&&pt>{}".format(1.093*pt,0.821*pt) 
      if("etapt" == cut):
        cutter="eta<1.&&eta>-1.&&pt<{}&&pt>{}".format(1.093*pt,0.821*pt) 
      zq.Draw("{}>>zqhist{}".format(varss[i],varss[i]),cutter)
      zg.Draw("{}>>zghist{}".format(varss[i],varss[i]),cutter)
      qq.Draw("{}>>qqhist{}".format(varss[i],varss[i]),cutter)
      gg.Draw("{}>>gghist{}".format(varss[i],varss[i]),cutter)

      mv=0
      for j in range(1,5):
        hists[i*5+j].Scale(1.0/hists[i*5+j].Integral())
        if(mv<hists[i*5+j].GetMaximum()):mv=hists[i*5+j].GetMaximum()
        hists[i*5+j].SetLineColor(j)
      #zghist.Scale(1.0/zghist.Integral())
      #qqhist.Scale(1.0/qqhist.Integral())
      #gghist.Scale(1.0/gghist.Integral())
      #if(mv<zghist.GetMaximum()):mv=zghist.GetMaximum()
      #if(mv<gghist.GetMaximum()):mv=gghist.GetMaximum()
      #if(mv<qqhist.GetMaximum()):mv=qqhist.GetMaximum()
      #zghist.SetLineColor(2)
      #qqhist.SetLineColor(3)
      #gghist.SetLineColor(4)
      hists[i*5].SetMaximum(mv*1.3)
      hists[i*5].Draw()
      for j in range(1,5):
        hists[i*5+j].Draw("Same")
      rt.gStyle.SetOptStat(False)
      rt.gPad.BuildLegend()
    canv.SaveAs("bdtvars/{}_{}.png".format(pt,cut))
    zqdata.Close()
    zgdata.Close()
    qqdata.Close()
    ggdata.Close()
