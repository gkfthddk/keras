from __future__ import division

from collections import OrderedDict
import numpy as np
import ROOT
from ROOT import TH1F,TColor
import os

# FIXME SetYTitle

class BinaryClassifierResponse(object):
    def __init__(self,
                 name,
                 title,
                 directory):

        self._name = name
        self._title = title
        self._directory = directory


        self._keys = [] 
        self._response = [] 
        self._path = os.path.join(directory, name + ".{ext}")

        self._test_sig_color = 38 # blue
        self._test_bkg_color = 46 # red
        print(self._response)

    def append(self, name, test):
        
        x, y = open("./save/{}/{}out.dat".format(name,test)).readlines()
        x, y= eval(x), eval(y)
        sig_response = x
        bkg_response = y
        if("cnn" in name):net="_CNN"
        if("rnn" in name):net="_RNN"
        if("v1t2" == test):event="Z+jet_"
        if("v1t3" == test):event="dijet_"
        sig_key = event + "test_signal"+net
        bkg_key = event + "test_background"+net
        self._keys.append(sig_key)
        self._keys.append(bkg_key)

        self._response.append(sig_response)
        self._response.append(bkg_response)

        
    def _draw(self):
        palette=[4,2,4,2,6,8,6,8]
        #palette=[866,857,822,814,624,634,798,807,886,873]
        self._palette=[palette[i] for i in range(len(self._keys))]

        canvas = ROOT.TCanvas("c", "c", 1200, 800)
        canvas.cd()

        h0 = TH1F("untitled", "", 50, 0, 1)
        h0.SetXTitle("Model response")
        h0.SetYTitle("Normalized")
        h0.GetXaxis().SetTitleSize(0.04)
        h0.GetYaxis().SetTitleSize(0.04)

        hists=[]
        for i in range(len(self._keys)):
            key=self._keys[i]
            hists.append(TH1F(key, key, 50, 0, 1))
            for each in self._response[i]:
                hists[i].Fill(each) 
                

        # Normalization
        for hist in hists:
            #for j in range(50):
            #  hist.SetBinError(j+1,1*(hist.GetBin(j+1)))
            hist.Scale(1.0 / hist.Integral())

        max_value = max(each.GetMaximum() for each in hists)
        h0.SetMaximum(1.4 * max_value)

        # Color
        for i in range(len(self._keys)):
            color=self._palette[i]
            #ROOT.TColor(color,*self._palette[i])
            print(color)
            
            hists[i].SetLineColor(color)
            #hists[i].SetMarkerStyle(21)
            if("dijet" in self._keys[i]):
                if(color==4):hists[i].SetMarkerStyle(20)
                if(color==2):hists[i].SetMarkerStyle(21)
            hists[i].SetMarkerSize(1.4)
            #if("back" in self._keys[i]):hists[i].SetFillColor(color)
            hists[i].SetMarkerColor(color)
            if("Z+jet" in self._keys[i]):
                hists[i].SetFillColorAlpha(color,0.5)
                if(color==4):hists[i].SetFillColorAlpha(38,0.6)
                if(color==2):hists[i].SetFillColorAlpha(46,0.6)#hists[i].SetFillStyle(3354)


        #hists["Z+jet_test_signal_RNN"].SetFillColorAlpha(self._test_sig_color, 0.333)
        #hists["Z+jet_test_background_RNN"].SetFillColor(self._test_bkg_color)
        # FIXME attribute
        #hists["Z+jet_test_background_RNN"].SetFillStyle(3354)

        # Draw
        h0.Draw("hist L")
        for i in range(len(self._keys)):
            if("dijet" in self._keys[i]):hists[i].Draw("EP same")
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
            label = "{}(pure {} sample)".format(cls.title(),event)
            option = "pl" if "dijet" in self._keys[i] else "lf"
            legend.AddEntry(hist, label, option)
        legend.Draw()


        ROOT.gStyle.SetOptStat(False)

        canvas.SaveAs(self._path.format(ext="png"))
        canvas.SaveAs(self._path.format(ext="pdf"))

pts=[100,200,500,1000]
for pt in pts:
    filename="plots/cnneventsvs{}".format(pt)
    a= BinaryClassifierResponse(filename,"{}~{}GeV".format(pt,int(pt*1.1)),"./")
    if("rnn" in filename):
      a.append("pepzjrnn{}sgd".format(pt),"v1t2")
      a.append("pepzjrnn{}sgd".format(pt),"v1t3")
    if("cnn" in filename):
      a.append("pepzjcnn{}model".format(pt),"v1t2")
      a.append("pepzjcnn{}model".format(pt),"v1t3")
    a._draw()
