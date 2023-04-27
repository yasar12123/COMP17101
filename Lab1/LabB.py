import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# convert the data into a data frame
boston_df = pd.read_csv("boston.csv")
pd.set_option("display.max.columns", None)
print(boston_df.describe())


# frequency of ZN graph
plt.figure()
sns.histplot(boston_df['ZN'], kde=True, stat="density", linewidth=0, kde_kws=dict(cut=3))
plt.xlabel('ZN')
plt.ylabel('Frequency')
plt.show()


# frequency logarithm graph for ZN
plt.figure()
sns.histplot(np.log1p(boston_df['ZN']), kde=True, stat="density", linewidth=0, kde_kws=dict(cut=3))
plt.xlabel('ZN')
plt.ylabel('Frequency')
plt.show()


# frequency of INDUS graph
plt.figure()
sns.histplot(boston_df['INDUS'], kde=True, stat="density", linewidth=0, kde_kws=dict(cut=3))
sns.rugplot(boston_df, x="INDUS")
plt.xlabel('INDUS')
plt.ylabel('Frequency')
plt.show()


# seaborns pairplot function
sns.pairplot(boston_df)
plt.show()


# heatmap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(boston_df.corr(), square=True, linewidths=.5,
            vmin=-1.0, vmax=1.0, cmap=cmap, annot=False)
plt.show()


