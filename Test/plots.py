import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


#Read the csv file
pre_df = pd.read_csv("Gemini_BTCUSD_1h.csv", header=1)
#datetime col
pre_df['datetime'] = pd.to_datetime(pre_df["date"])
#sorted df by datetime
df = pre_df.sort_values(by='datetime', ascending=True)



#plots
plt.figure(1, figsize=(16,6))
plt.plot(df['close'])
plt.show()


pd.set_option("display.max.columns", None)
print(df.head())
