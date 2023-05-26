import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

#Read the csv file
dfRaw = pd.read_csv("Gemini_BTCUSD_1h.csv", header=1)

#remove columns
dfRaw.drop('unix', axis=1, inplace=True)
dfRaw.drop('symbol', axis=1, inplace=True)

# convert the 'date' column to datetime format
dfRaw['datetime'] = pd.to_datetime(dfRaw["date"])

#sort by date
df_sorted = dfRaw.sort_values(by=["date"], ascending=True)
#reset index
df = df_sorted.reset_index(drop=True)

#df = df[:100]
#create date features
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



#print(df[['date', 'close', 'PrevClose', 'Log Return','BullishBearish']])


#sns.lineplot(x='date', y='close', data=df)
sns.lineplot(x='Hour', y='BullishBearish', data=df)
plt.show()


