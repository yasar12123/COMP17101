import pandas as pd

import numpy as np
import pandas as pd
import pandas_ta as ta
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


#Read the csv file
dfRaw = pd.read_csv('GBPUSD_D1.csv', sep=r"\t")

# convert the 'date' column to datetime format
dfRaw['datetime'] = pd.to_datetime(dfRaw["Time"])

#sort by date
dfSorted = dfRaw.sort_values(by=["datetime"], ascending=True)
#reset index
dfSorted = dfSorted.reset_index(drop=True)
#set datetime as index
#dfSorted = dfSorted.set_index('datetime')


# convert the 'date' column to datetime format
dfSorted['Date'] = pd.to_datetime(dfSorted["datetime"])
dfSorted['Year'] = dfSorted['Date'].dt.year
dfSorted['Month'] = dfSorted['Date'].dt.month
dfSorted['Week'] = dfSorted['Date'].dt.isocalendar().week
dfSorted['DayOfWeek'] = dfSorted['Date'].dt.dayofweek
dfSorted['DayOfMonth'] = dfSorted['Date'].dt.day

#
dfSorted['LogReturn'] = np.log(dfSorted['Close'].shift(1)/dfSorted['Close']) * 100

#next n days close price log
dfSorted['CloseN+3Log'] = (np.log(dfSorted['Close'].shift(-3) / dfSorted['Close'])) * 100


#if percentage increase is greater than 0.25% then flag as 3 (bullish)
#if percentage decrease is less than -0.25 then flag as 2 (bearish)
#else 1 (neutral)
#define conditions
conditions = [ (dfSorted['CloseN+3Log'] >= -0.5) & (dfSorted['CloseN+3Log'] <= 0.5), # 0 - neutral
               (dfSorted['CloseN+3Log'] > 0.5), # 1 - uptrend
               (dfSorted['CloseN+3Log'] < -0.5) # 2 -  downtrend
              ]
#define results
results = [0,1,2]
#create feature
dfSorted['BullishBearish'] = np.select(conditions, results)


#TA indicators
dfSorted['RSI14'] = ta.rsi(dfSorted['Close'], 14)
dfSorted['EMA200'] = ta.ema(dfSorted['Close'], 200)
dfSorted['EMA100'] = ta.ema(dfSorted['Close'], 100)
dfSorted['EMA50'] = ta.ema(dfSorted['Close'], 50)
dfSorted.ta.stoch(high='high', low='low', k=14, d=3, append=True)
dfSorted.ta.macd(close='close', append=True)
dfSorted.ta.adx(close='close', append=True)


#drop all nan values
dfSorted = dfSorted.dropna()

#view all columns
pd.set_option("display.max.columns", None)
print(dfSorted)
print(dfSorted['BullishBearish'].value_counts())


#plots
# fig, ax = plt.subplots()
# dfSorted.groupby("BullishBearish").plot(x="datetime", y="Close", marker="o", ax=ax)
# ax.legend(['a','b','c'])
# plt.show()

# sns.boxplot( x=dfSorted['BullishBearish'], y=dfSorted['CloseN+3Log'] )
# plt.show()

#heatmap correlation
# corr_matrix = dfSorted.corr(method='spearman')
# f, ax = plt.subplots(figsize=(16,8))
# sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidth=0.4,
#             annot_kws={"size": 10}, cmap='coolwarm', ax=ax)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.show()

#plot moving averages
# plt.plot(dfSorted['Date'], dfSorted['Close'], label="close")
# plt.plot(dfSorted['Date'], dfSorted['EMA200'], label="EMA200")
# plt.plot(dfSorted['Date'], dfSorted['EMA100'], label="EMA100")
# plt.plot(dfSorted['Date'], dfSorted['EMA50'], label="EMA50")
# plt.legend()
# plt.show()

#distribution
# dfSorted['Volume'].plot()
# plt.show()


#close trend with buy, sell, hold
# fig, ax = plt.subplots()
# dfSorted.groupby("BullishBearish").plot(x="datetime", y="Close", marker="o", ax=ax, linestyle='none')
# plt.plot(dfSorted['datetime'], dfSorted['Close'], label="close")
# ax.legend(['Hold','Buy','Sell'])
# plt.show()
#

