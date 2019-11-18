import numpy as np
import ROOT as rt
import sys
f=rt.TFile(sys.argv[1],"read")
jet=f.Get("jetAnalyser")
jet.GetEntry(470)
print(np.array(jet.image_chad_mult_25).max())
