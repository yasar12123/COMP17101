import numpy as np
import pandas as pd
import pandas_ta as ta
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeClassifier

#Read the csv file
dfRaw = pd.read_csv("Gemini_BTCUSD_1h.csv", header=1)

# convert the 'date' column to datetime format
dfRaw['datetime'] = pd.to_datetime(dfRaw["date"])

#sort by date
df_sorted = dfRaw.sort_values(by=["datetime"], ascending=True)
#reset index
df = df_sorted.reset_index(drop=True)

#remove columns
dfRaw.drop('unix', axis=1, inplace=True)
dfRaw.drop('symbol', axis=1, inplace=True)
dfRaw.drop('date', axis=1, inplace=True)

#df = df[:100]
#create date features
df['Date'] = df['datetime'].dt.date
df['Year'] = df['datetime'].dt.year
df['Month'] = df['datetime'].dt.month
df['Week'] = df['datetime'].dt.isocalendar().week
df['DayOfWeek'] = df['datetime'].dt.dayofweek
df['Day'] = df['datetime'].dt.day
df['Hour'] = df['datetime'].dt.hour
#misc
df['PrevClose'] = df['close'].shift(1)
df['Log Return'] = np.log(df['PrevClose']/df['close'])
df['BullishBearish'] = df['Log Return'].apply(lambda x: 1 if x > 0 else 0)





#TA indicators
df['rsi'] = ta.rsi(df['open'], 14)
df['rsiBelow20'] = df['rsi'].apply(lambda x: 1 if x <= 20 else 0)
df['rsiAbove70'] = df['rsi'].apply(lambda x: 1 if x >= 80 else 0)

rsiBelow20 = df[df['rsiBelow20'] == 1]
rsiAbove70 = df[df['rsiAbove70'] == 1]
a = rsiBelow20.groupby('BullishBearish')['rsiBelow20'].count()
b = rsiAbove70.groupby('BullishBearish')['rsiAbove70'].count()
print('Bearish day - rsi below 20: {}'.format(a[0]),'\n'
      'Bullish day - rsi below 20: {}'.format(a[1]))
print('Bearish day - rsi above 70: {}'.format(b[0]),'\n'
      'Bullish day - rsi above 70: {}'.format(b[1]))


#daily open, close, high and low
#dailyDf = df[['Date','Hour','open','close','high','low']]


#print(df[['Date', 'close', 'PrevClose', 'Log Return','BullishBearish']])

#sns.lineplot(x='date', y='rsi', data=df)
#sns.lineplot(df['rsiAbove70'])
#sns.lineplot(df['Log Return'])
sns.lineplot(df['close'])
plt.show()


