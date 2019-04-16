import numpy as np
import matplotlib.pyplot as plt
fs=25
plt.figure(figsize=(12, 8))
plt.ylabel("ROC AUC",fontsize=fs*1.3)
plt.xlabel("$p_T$ Range(GeV)",fontsize=fs*1.3)
plt.plot([105,210,525,1050],[0.815,0.833,0.846,0.841],
          ':',linewidth=3,label="BDT-Z+jet",marker='o',
          alpha=0.7,color='C2',markersize=fs)
plt.plot([105,210,525,1050],[0.828,0.849,0.853,0.823],
          ':',linewidth=3,label="CNN-Z+jet",marker='D',
          alpha=0.7,color='C0',markersize=fs*0.75)
plt.plot([105,210,525,1050],[0.817,0.837,0.835,0.807],
          ':',linewidth=3,label="RNN-Z+jet",marker='^',
          alpha=0.7,color='C1',markersize=fs)
plt.plot([105,210,525,1050],[0.811,0.827,0.832,0.825],
          ':',linewidth=3,label="BDT-Dijet",marker='o',
          fillstyle='none',color='C2',markersize=fs)
plt.plot([105,210,525,1050],[0.826,0.833,0.819,0.797],
          ':',linewidth=3,label="CNN-Dijet",marker='D',
          fillstyle='none',color='C0',markersize=fs*0.75)
plt.plot([105,210,525,1050],[0.811,0.819,0.796,0.779],
          ':',linewidth=3,label="RNN-Dijet",marker='^',
          fillstyle='none',color='C1',markersize=fs)
plt.xticks([105,210,525,1050],["100\n~110","200\n~220","500\n~550","1000\n~1100"],size=fs*0.8)
plt.yticks([0.76,0.78,0.8,0.82,0.84,0.86],size=fs)
plt.grid(alpha=0.6)
plt.legend(fontsize=fs*0.8,ncol=2)
plt.savefig("realistic.pdf",bbox_inches='tight',pad_inches=0.5,dpi=300)
plt.savefig("realistic.png",bbox_inches='tight',pad_inches=0.5,dpi=300)
plt.figure(figsize=(12, 8))
plt.ylabel("ROC AUC",fontsize=fs*1.3)
plt.xlabel("$p_T$ Range(GeV)",fontsize=fs*1.3)
plt.plot([105,210,525,1050],[0.85,0.87,0.889,0.894],
          ':',linewidth=3,label="CNN-pure",marker='D',
          color='C0',fillstyle='none',markersize=fs*0.75)
plt.plot([105,210,525,1050],[0.846,0.857,0.866,0.868],
          ':',linewidth=3,label="RNN-pure",marker='^',
          color='C1',fillstyle='none',markersize=fs)
plt.plot([105,210,525,1050],[0.828,0.849,0.853,0.823],
          ':',linewidth=3,label="CNN-realistic",marker='D',
          alpha=0.7,color='C0',markersize=fs*0.75)
plt.plot([105,210,525,1050],[0.817,0.837,0.835,0.807],
          ':',linewidth=3,label="RNN-realistic",marker='^',
          alpha=0.7,color='C1',markersize=fs)
plt.xticks([105,210,525,1050],["100\n~110","200\n~220","500\n~550","1000\n~1100"],size=fs*0.8)
plt.yticks([0.8,0.82,0.84,0.86,0.88,0.9],size=fs)
plt.grid(alpha=0.6)
plt.legend(fontsize=fs*0.8,ncol=2,loc=8)
plt.savefig("pure.pdf",bbox_inches='tight',pad_inches=0.5,dpi=300)
plt.savefig("pure.png",bbox_inches='tight',pad_inches=0.5,dpi=300)
