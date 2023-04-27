import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# convert the data into a data frame
boston_df = pd.read_csv("boston.csv")
pd.set_option("display.max.columns", None)
#print(boston_df.describe())



sns.heatmap(boston_df['CRIM'])
plt.show()