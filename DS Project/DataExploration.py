import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import preProcessData


#Read the csv file
dfRaw = preProcessData.dfDaily #pd.read_csv("Gemini_BTCUSD_1h.csv", header=1)

#to view all columns
pd.set_option("display.max.columns", None)

print(dfRaw.head())
print(dfRaw.tail())
print(dfRaw.describe())
print(dfRaw.info())
print(dfRaw.std(numeric_only=None))

#plots
#sns.boxplot(x='BullishBearish', y='rsi', data=dfRaw)
#sns.boxplot(dfRaw['close'])
#sns.lineplot(dfRaw[['open', 'close']])
#plt.show()

#heatmap correlation
corr_matrix = dfRaw.corr(method='spearman')
f, ax = plt.subplots(figsize=(16,8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidth=0.4,
            annot_kws={"size": 10}, cmap='coolwarm', ax=ax)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()



