import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
fs=25
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
        x ,y= open("save/{}/{}roc.dat".format(name,test)).readlines()
        x, y= np.array(eval(x)), np.array(eval(y))
        x=[x[i] for i in range(1,len(x),2)]
        y=[y[i] for i in range(1,len(y),2)]
        self.data_list.append({"x": x, "y": y,"name":name,"rat":rat,"test":test, "auc": auc(x, y),"col":color,"leg":legend})
    def get_ens_result(self,name,rat,test,color=None,legend=None):
        x ,y= open("ensemble/{}{}{}roc.dat".format(name,rat,test)).readlines()
        x, y= eval(x), eval(y)
        try:self.data_list.append({"x": x, "y": y,"name":name,"rat":eval(rat),"test":test, "auc": auc(x, y),"col":color})
        except:self.data_list.append({"x": x, "y": y,"name":name,"rat":0,"test":test, "auc": auc(x, y),"col":color,"leg":legend})
    def get_tm_result(self,name,rat,test,color=None,legend=None):
        x= open("../tmva/{}roc.txt".format(name.format(int(rat*100)))).readline()
        x= eval(x)
        y=np.arange(500.)/500
        self.data_list.append({"x": x, "y": y,"name":name,"rat":rat,"test":test, "auc": auc(x, y),"col":color,"leg":legend})
        
    def sort_data_list(self):
        self.data_list = sorted(self.data_list, key=lambda data: -1.*data["auc"])
        
    def compare_roc(self,title="Quark/Gluon Jet Dicrimination", filename=None):
        from sklearn import metrics
    
        
        plt.figure(figsize=(12, 8))
        #plt.title(title,fontdict={"weight": "bold", "size": 22})
        plt.xlabel("Quark Jet Efficiency", fontsize=fs*1.2)
        plt.ylabel("Gluon Jet Rejection", fontsize=fs*1.2)
        plt.tick_params(labelsize=fs)
        # data = {"x": sig_eff, "y": bkg_rej, "label": leg_str, "auc": auc_num}
        cmap = get_cmap(len(self.data_list))
        for idx, data in enumerate(self.data_list):
            if(data["col"]!=None):
              col=data["col"]
            else:
              col=cmap(idx)
            if("CNN" in data['leg']):
              ls="-"
              if("1000" in data['leg']):ls="-"
            if("RNN" in data['leg']):
              ls="--"
              if("1000" in data['leg']):ls="--"
            #if(data['leg']):ls="--"
            plt.plot(data["x"], data["y"],label="{}".format(data["leg"],auc=data["auc"]), lw=4, alpha=0.75,color=col,linestyle=ls)
            print(data["leg"],data["auc"])
        
        #plt.legend(loc="lower left", fontsize=20)
        plt.legend(loc=3, fontsize=fs*0.9)
        plt.grid(alpha=0.6)
        #plt.xscale('log')
        #plt.yscale('log')
        a1,a2,b1,b2=plt.axis()
        plt.axis((0,1,0,1))
        plt.savefig(filename,bbox_inches='tight',dpi=300,pad_inches=0.5)
        plt.savefig(filename+".pdf",bbox_inches='tight',dpi=300,pad_inches=0.5)
            
    def compare_AUC(self,title="Quark/Gluon Jet Dicrimination", color=None, filename=None):
        from sklearn import metrics   
        plt.figure(figsize=(12, 8))
        plt.grid(True)
        plt.title(title,
                  fontdict={"weight": "bold", "size": 22})
        plt.xlabel("pt", fontsize=22)
        plt.ylabel("AUC", fontsize=22)
        plt.tick_params(labelsize=22)
        # data = {"x": sig_eff, "y": bkg_rej, "label": leg_str, "auc": auc_num}
        aucname=[]
        aucdata=[]
        for idx, data in enumerate(self.data_list):
            name=""
            print(data["name"]+data['test'])
            if('v1' in data["name"]+data['test']):name+="generic-"
            if('v2' in data["name"]+data['test']):name+="Z+jet-"
            if('v3' in data["name"]+data['test']):name+="dijet-"
            if('t2' in data["name"]+data['test']):name+="Z+jet"
            if('t3' in data["name"]+data['test']):name+="dijet"
            if('cnn' in data["name"]+data['test']):name+=" CNN"
            if('rnn' in data["name"]+data['test']):name+=" RNN"
            print(name)

            if(aucname.count(name)==0):
                aucname.append(name)
                aucdata.append([[],[],[],[]])
                iii=len(aucname)-1
            else:
                iii=aucname.index(name)
            aucdata[iii][0].append(data["rat"])
            aucdata[iii][1].append(data["auc"])
            aucdata[iii][2]=data["col"]
            aucdata[iii][3]=name
        print len(aucname)
        cmap = get_cmap(len(aucname))
        print(aucdata)
        aucdata=sorted(aucdata,key=lambda x:-x[1][-1])

        for i in range(len(aucname)):
            color=cmap(len(aucname)-i-1)
            line="-"
            if(aucdata[i][2]!=None):color=aucdata[i][2]
            if("RNN" in aucdata[i][3]): line="--"
            
            plt.plot(aucdata[i][0], aucdata[i][1],line, marker="o",label="{}".format(aucdata[i][3]), lw=3, alpha=0.5,color=color)
            
        a1,a2,b1,b2=plt.axis()
        #plt.xticks(np.arange(0.5,1.1,step=0.05))
        #plt.axis((0.55-0.0225,1.0+0.0225,0.68,0.85))
        print a1,a2, aucdata[0][0]
        #plt.gca().invert_xaxis() 
        #plt.legend(loc="lower left", fontsize=20)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=22)
        #plt.tight_layout()
        print("filename",filename)
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
events=["zj"]
nets=["cnn","rnn"]
pts=["100"]
#pts=["200"]
nalist=["Z+j,jj","Z+q,Z+g","qq,gg"]
#pt=pts[0]
dlroc = CompareROC()
for pt in pts:
  #dlroc = CompareROC()
  #dlout=CompareOUT()
  for event in events:
    for net in nets:
      name="pep"+event+net+pt+"sgd631"
      if(net=="cnn"):name="pep"+event+net+pt+"model"
      #print(name)
      if(event=="zj"):
        dlroc.get_yj_result(name,pt,"v1t2",legend="{}-Z+jet($p_T$ {}~{}GeV)".format(net.upper(),pt,int(eval(pt)*1.1)),color="black")
        dlroc.get_yj_result(name,pt,"v1t3",legend="{}-dijet($p_T$ {}~{}GeV)".format(net.upper(),pt,int(eval(pt)*1.1)),color="orangered")
      if(event=="zq"):
        dlroc.get_yj_result(name,pt,"v2t2",legend="pure-Z+jet "+net.upper()+pt,color=None)
      if(event=="qq"):
        dlroc.get_yj_result(name,pt,"v3t3",legend="pure-dijet "+net.upper()+pt,color=None)

  #plotname="plots/pepzq"+pt+"mix"
  #dlroc.compare_roc("ROC curve",filename=plotname+"roc")
  #dlout.compare_out("Output distribution",filename=plotname+"out")
#dlroc.sort_data_list()
plotname="roc100"
#plotname="plots/pepzqrnn"
print(plotname)
dlroc.compare_roc("ROC curve",filename=plotname)


