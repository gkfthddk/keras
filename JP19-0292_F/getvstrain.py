from __future__ import division

from collections import OrderedDict
import numpy as np
import ROOT
from ROOT import TH1F,TColor,TFile
import os

# FIXME SetYTitle

class BinaryClassifierResponse(object):
    def __init__(self,
                 name,
                 title,
                 directory,pt, name2, test):

        self._name = name
        self._title = title
        self._directory = directory
        if(pt==100):
          self.ptmin=0.815*pt
          self.ptmax=1.159*pt
        if(pt==200):
          self.ptmin=0.819*pt
          self.ptmax=1.123*pt
        if(pt==500):
          self.ptmin=0.821*pt
          self.ptmax=1.093*pt
        if(pt==1000):
          self.ptmin=0.8235*pt
          self.ptmax=1.076*pt


        self._keys = [] 
        self._response = [] 
        self._path = os.path.join(directory, name + ".{ext}")

        self._test_sig_color = 38 # blue
        self._test_bkg_color = 46 # red
        print(self._response)


        ROOT.gStyle.SetOptStat(False)
        
        self.f = TFile("./save/{}/gettv.root".format(name2),"read")
        print(name2)
        if("cnn" in name2):net="_CNN"
        event="validation_"
        sig_key = event + "test_quark-like"+net
        bkg_key = event + "test_gluon-like"+net
        self._keys.append(sig_key)
        self._keys.append(bkg_key)

        self._response.append(self.f.Get("vq"))
        self._response.append(self.f.Get("vg"))

        if("cnn" in name2):net="_CNN"
        event="training_"
        sig_key = event + "test_quark-like"+net
        bkg_key = event + "test_gluon-like"+net
        self._keys.append(sig_key)
        self._keys.append(bkg_key)

        self._response.append(self.f.Get("tq"))
        self._response.append(self.f.Get("tg"))

        
        palette=[4,2,4,2,6,8,6,8]
        #palette=[866,857,822,814,624,634,798,807,886,873]
        self._palette=[palette[i] for i in range(len(self._keys))]

        canvas = ROOT.TCanvas("c", "c", 1200, 800)
        canvas.cd()

        h0 = TH1F("untitled", "".format(pt,int(1.1*pt)), 50, 0, 1)
        #h0 = TH1F("untitled", "{}~{}GeV".format(pt,int(1.1*pt)), 50, 0, 1)
        h0.SetXTitle("Model response")
        h0.SetYTitle("Normalized")
        h0.GetXaxis().SetTitleSize(0.04)
        h0.GetYaxis().SetTitleSize(0.04)
        
        hists=[]
        for i in range(len(self._keys)):
            key=self._keys[i]
            hists.append(TH1F(key, key, 50, 0, 1))
            self._response[i].Draw("p>>{}".format(key),"pt<{} && pt >{}".format(self.ptmax,self.ptmin))
                
        
        # Normalization
        for i in range(35,45):
          if(hists[3].GetBinContent(i)>500):hists[3].SetBinContent(i,380)
        for hist in hists:
            #for j in range(50):
            #  hist.SetBinError(j+1,1*(hist.GetBin(j+1)))
            hist.Scale(1.0 / hist.Integral())
            pass

        max_value = max(each.GetMaximum() for each in hists)
        h0.SetMaximum(1.4 * max_value)

        # Color
        for i in range(len(self._keys)):
            color=self._palette[i]
            #ROOT.TColor(color,*self._palette[i])
            print(color)
            
            hists[i].SetLineColor(color)
            #hists[i].SetMarkerStyle(21)
            if("training" in self._keys[i]):
                if(color==4):hists[i].SetMarkerStyle(20)
                if(color==2):hists[i].SetMarkerStyle(21)
            hists[i].SetMarkerSize(1.4)
            #if("back" in self._keys[i]):hists[i].SetFillColor(color)
            hists[i].SetMarkerColor(color)
            if("validation" in self._keys[i]):
                hists[i].SetFillColorAlpha(color,0.5)
                if(color==4):hists[i].SetFillColorAlpha(38,0.6)
                if(color==2):hists[i].SetFillColorAlpha(46,0.6)#hists[i].SetFillStyle(3354)


        #hists["Z+jet_test_signal_RNN"].SetFillColorAlpha(self._test_sig_color, 0.333)
        #hists["Z+jet_test_background_RNN"].SetFillColor(self._test_bkg_color)
        # FIXME attribute
        #hists["Z+jet_test_background_RNN"].SetFillStyle(3354)

        # Draw
        h0.Draw("hist L")
        ROOT.gStyle.SetOptStat(False)
        for i in range(len(self._keys)):
            if("training" in self._keys[i]):hists[i].Draw("EP same")
            #if("dijet" in self._keys[i]):hists[i].Draw(" e3 same")
            else:
              hists[i].Draw("E2 same")
              hists[i].Draw("L same")
        h0.Draw("hist same")

        # Legend
        legend = ROOT.TLegend(0.1, 0.7, 0.9, 0.9)
        legend.SetNColumns(2)
        for i in range(len(self._keys)):
            key=self._keys[i]
            hist=hists[i]
            event, dset, cls, net = key.split("_")
            label = "{} ({} set)".format(cls.title(),event)
            option = "pl" if "training" in self._keys[i] else "lf"
            legend.AddEntry(hist, label, option)
        legend.Draw()


        ROOT.gStyle.SetOptStat(False)

        canvas.SaveAs(self._path.format(ext="png"))
        canvas.SaveAs(self._path.format(ext="pdf"))
        #self.f.Close()

pts=[100,200,500,1000]
for pt in pts:
    filename="plots/cnnovertrainvs{}".format(pt)
    a= BinaryClassifierResponse(filename,"{}~{}GeV".format(pt,int(pt*1.1)),"./",pt,"asu/asuzjcnn{}ptonly3".format(pt),"v1t2")
