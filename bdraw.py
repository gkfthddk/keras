import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import auc
def hist(data,bi=30):
    hi=np.zeros(bi)
    for i in data:
        hi[int(np.floor(i*bi))]+=1
    hi=bi*hi/len(data)
    return hi
class CompareROC(object):
    def __init__(self):
        self.data_list = []
        
    def get_result(self, path, label,color=None):
        data = np.loadtxt(path, delimiter=",", unpack=True)
        x = data[0]
        y = data[1]
        self.data_list.append({"x": x, "y": y, "label": label, "auc": auc(x, y),"col":color})
        
    def get_yj_result(self,name,rat,test,color=None,legend=None):
        x ,y= open("save/{}{}/{}roc.dat".format(name,rat,test)).readlines()
        x, y= eval(x), (1-np.array(eval(y))).tolist()
        aucv=auc(x,(1-np.array(y)).tolist())
        bkg=[]
        sig=[]
        for efq, efg in zip(x,y):
            if(efq<0.1 or efg==0):continue
            sig.append(efq)
            bkg.append(1/efg)
        y=bkg 
        x=sig
        try:self.data_list.append({"x": x, "y": y,"name":name,"rat":eval(rat),"test":test, "auc": auc(x, y),"col":color})
        except:self.data_list.append({"x": x, "y": y,"name":name,"rat":0,"test":test, "auc": aucv,"col":color,"leg":legend})
    
    def sort_data_list(self):
        self.data_list = sorted(self.data_list, key=lambda data: -1.*data["auc"])
        
    def compare_roc(self,title="Quark/Gluon Jet Dicrimination", filename=None):
        from sklearn import metrics
    
        
        plt.figure(figsize=(12, 9))
        plt.grid(True)
        plt.title(title,
                  fontdict={"weight": "bold", "size": 22})
        plt.xlabel("Signal efficiency", fontsize=22)
        plt.ylabel("1 / background efficiency", fontsize=22)
        plt.tick_params(labelsize=22)
        # data = {"x": sig_eff, "y": bkg_rej, "label": leg_str, "auc": auc_num}
        cmap = get_cmap(len(self.data_list))
        for idx, data in enumerate(self.data_list):
            if("t2" in data["test"]):ls='-'
            else:ls='--'
            if(data["col"]!=None):
                plt.plot(data["x"], data["y"],ls,label="{}".format(data["leg"],auc=data["auc"]), lw=3, alpha=0.7,color=data["col"])
            else:
                plt.plot(data["x"], data["y"],ls,label="{}".format(data["leg"],auc=data["auc"]), lw=3, alpha=0.7,color=cmap(idx))
                  #plt.plot(data["x"], data["y"],label="{}(AUC={auc:.3f})".format(data["leg"],auc=data["auc"]), lw=3, alpha=0.5,color=cmap(len(self.data_list)-idx-1))
            print(data["leg"],data["auc"])
        
        plt.legend(bbox_to_anchor=(0.999,0.998),loc=1, borderaxespad=0.,fontsize=22,framealpha=1)
        #plt.legend(bbox_to_anchor=(0.,0.1), loc=2, borderaxespad=0.,fontsize=22)
        a1,a2,b1,b2=plt.axis()
        plt.axis((0.1,1,1,b2))
        plt.yscale('log')
        if filename is not None:
            plt.savefig(filename,bbox_inches='tight',dpi=100)
            
    def compare_AUC(self,title="Quark/Gluon Jet Dicrimination", color=None, filename=None):
        from sklearn import metrics   
        plt.figure(figsize=(12, 9))
        plt.grid(True)
        plt.title(title,
                  fontdict={"weight": "bold", "size": 22})
        plt.xlabel("fraction", fontsize=22)
        plt.ylabel("AUC", fontsize=22)
        plt.tick_params(labelsize=22)
        # data = {"x": sig_eff, "y": bkg_rej, "label": leg_str, "auc": auc_num}
        aucname=[]
        aucdata=[]
        for idx, data in enumerate(self.data_list):
            if(aucname.count(data["name"]+"-"+data["test"])==0):
                aucname.append(data["name"]+"-"+data["test"])
                aucdata.append([[],[],[]])
                iii=len(aucname)-1
            else:
                iii=aucname.index(data["name"]+"-"+data["test"])
            aucdata[iii][0].append(data["rat"])
            aucdata[iii][1].append(data["auc"])
            aucdata[iii][2]=data["col"]
        print len(aucname)
        cmap = get_cmap(len(aucname))
        for i in range(len(aucname)):
            if(aucdata[i][2]!=None):
                plt.plot(aucdata[i][0], aucdata[i][1],"-", marker="o",label="{}".format(aucname[i]), lw=3, alpha=0.5,color=aucdata[i][2])
            else:
                plt.plot(aucdata[i][0], aucdata[i][1],"-", marker="o",label="{}".format(aucname[i]), lw=3, alpha=0.5,color=cmap(len(aucname)-i-1))
        a1,a2,b1,b2=plt.axis()
        plt.xticks(np.arange(0.5,1.1,step=0.05))
        plt.axis((0.55-0.0225,1.0+0.0225,0.68,0.85))
        print a1,a2, aucdata[0][0]
        #plt.gca().invert_xaxis() 
        #plt.legend(loc="lower left", fontsize=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=22)
        #plt.tight_layout()
        if filename is not None:
            plt.savefig(filename,bbox_inches='tight',dpi=100)
    def clear_data_list(self):
        self.data_list = []
def get_cmap(n, name='brg'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
names=[]
nalist=[]
nalabel=[]
events=["zj","zq","qq"]
nets=["cnn"]
#nets=["rnn","cnn"]
pts=["200","1000"]
#pts=["100","200","500","1000"]
nalist=["Z+j,jj","Z+q,Z+g","qq,gg"]
#pt=pts[0]
dlroc = CompareROC()
for pt in pts:
  #dlroc = CompareROC()
  for event in events:
    for net in nets:
      name="pep"+event+net+pt+"sgd"
      if(net=="cnn"):name="pep"+event+net+pt+"model"
      #print(name)
      if(event=="zj"):
        dlroc.get_yj_result(name,"","v1t2",legend="Z+jet($p_T$ {}~{})".format(pt,int(eval(pt)*1.1)))
        dlroc.get_yj_result(name,"","v1t3",legend="dijet($p_T$ {}~{})".format(pt,int(eval(pt)*1.1)))
      #if(event=="zq"):
      #  dlroc.get_yj_result(name,"","v2t2",legend="Z+jet-Z+jet "+"{}~{}".format(pt,int(eval(pt)*1.1)))
      #if(event=="qq"):
      #  dlroc.get_yj_result(name,"","v3t3",legend="dijet-dijet "+"{}~{}".format(pt,int(eval(pt)*1.1)))

  #dlroc.sort_data_list()
  plotname="plots/pepzj"+pt+"mix"
plotname="plots/pepcnn"
print(plotname)
dlroc.compare_roc("",filename=plotname+"1bkg")
