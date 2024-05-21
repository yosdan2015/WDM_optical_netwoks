import pandas as pd
import numpy 
import math
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("C:/Users/Yosdan/OneDrive/Glendo/Valeska/Optisystem/data_valeska.csv")
variaveis=["Power_WDM","EDFA1","EDFA2","EDFA3","Total_Gain","Total_Power","Min_BER","Max_Q_Factor"]

dados_jones=df[variaveis]



matrix_correlacao_spearman=round(dados_jones.corr(method="spearman"),2)
matrix_correlacao_pearson=round(dados_jones.corr(method="pearson"),2)
matrix_correlacao_kendall=round(dados_jones.corr(method="kendall"),2)

# Se o coeficiente de correlação for muito próximo de zero, trata como zero




fig, (ax1, ax2, ax3) = plt.subplots(ncols=3,nrows=1)


sns.heatmap(matrix_correlacao_pearson,  cmap="coolwarm", annot=True, 
            cbar=False, vmin=-1, vmax=1, center=0,linewidths=0.5,
            xticklabels=True, yticklabels=True, square=True, ax=ax1)
sns.heatmap(matrix_correlacao_spearman,  cmap="coolwarm", annot=True, 
            cbar=False, vmin=-1, vmax=1, center=0,linewidths=0.5,
            xticklabels=True, yticklabels=True, square=True, ax=ax2)
sns.heatmap(matrix_correlacao_kendall,  cmap="coolwarm", annot=True, 
            cbar=False, vmin=-1, vmax=1, center=0,linewidths=0.5,
            xticklabels=True, yticklabels=True, square=True, ax=ax3)

ax1.set_title("Pearson")
ax2.set_title("Spearman")
ax3.set_title("Kendall")

plt.show()
