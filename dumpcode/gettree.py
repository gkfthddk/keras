import ROOT as rt
from array import array
import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--save",type=str,default="ten100grucnn",help='rch')
parser.add_argument("--pt",type=int,default=100,help='pt range pt~pt*1.1')
args=parser.parse_args()
pt=args.pt
savename="save/"+str(args.save)
dqfile=[eval(i) for i in open("{}/v1t3q.dat".format(savename)).readlines()]
zqfile=[eval(i) for i in open("{}/v1t2q.dat".format(savename)).readlines()]
dgfile=[eval(i) for i in open("{}/v1t3g.dat".format(savename)).readlines()]
zgfile=[eval(i) for i in open("{}/v1t2g.dat".format(savename)).readlines()]
f=rt.TFile("{}/get.root".format(savename),"recreate")
dq=rt.TTree("dq","dq tree")
dg=rt.TTree("dg","dg tree")
zq=rt.TTree("zq","zq tree")
zg=rt.TTree("zg","zg tree")
p=array('f',[0.])
pt=array('f',[0.])
eta=array('f',[0.])
dq.Branch("p",p,"p/F")
dq.Branch("pt",pt,"pt/F")
dq.Branch("eta",eta,"eta/F")
dg.Branch("p",p,"p/F")
dg.Branch("pt",pt,"pt/F")
dg.Branch("eta",eta,"eta/F")
zq.Branch("p",p,"p/F")
zq.Branch("pt",pt,"pt/F")
zq.Branch("eta",eta,"eta/F")
zg.Branch("p",p,"p/F")
zg.Branch("pt",pt,"pt/F")
zg.Branch("eta",eta,"eta/F")
for i in range(len(dqfile[0])):
  p[0]=dqfile[0][i]
  pt[0]=dqfile[1][i]
  eta[0]=dqfile[2][i]
  dq.Fill()
for i in range(len(dgfile[0])):
  p[0]=dgfile[0][i]
  pt[0]=dgfile[1][i]
  eta[0]=dgfile[2][i]
  dg.Fill()
for i in range(len(zqfile[0])):
  p[0]=zqfile[0][i]
  pt[0]=zqfile[1][i]
  eta[0]=zqfile[2][i]
  zq.Fill()
for i in range(len(zgfile[0])):
  p[0]=zgfile[0][i]
  pt[0]=zgfile[1][i]
  eta[0]=zgfile[2][i]
  zg.Fill()
f.Write()
f.Close()
