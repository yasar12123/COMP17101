import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from datetime import datetime, date, timedelta

#Read the csv file
pre_df = pd.read_csv("Bitstamp_BTCUSD_1h.csv", header=1)
#datetime col
pre_df['datetime'] = pd.to_datetime(pre_df["date"])
#sorted df by datetime
df = pre_df.sort_values(by='datetime', ascending=True)

#set index
df.set_index('date')

#remove columns
df.drop('unix', axis=1, inplace=True)
df.drop('symbol', axis=1, inplace=True)

#create date features
df['Year'] = df['datetime'].dt.year
df['Month'] = df['datetime'].dt.month
df['Week'] = df['datetime'].dt.isocalendar().week
df['DayOfWeek'] = df['datetime'].dt.dayofweek
df['Day'] = df['datetime'].dt.day
df['Hour'] = df['datetime'].dt.hour

#remove columns
df.drop('datetime', axis=1, inplace=True)

#percentage of change of open and close price
df['PercChangeOpenClose'] = (df['close'] / df['open']) - 1

#calc column - if close price is greater than open then bullish otherwise bearish
#df['BullBear'] = np.where(df['PercChangeOpenClose'] >= 0, 1, 0)

#log
df['log'] = np.log1p(df['PercChangeOpenClose'])


# plots

# plt.figure(1, figsize=(16,6))
# plt.plot(df['PercChangeOpenClose'])
# plt.show()
#
# plt.figure(1, figsize=(16,6))
# plt.plot(df['log'])
# plt.show()



#train = df[['BullBear']]



#pd.set_option("display.max.columns", None)
print(df)