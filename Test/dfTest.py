import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator


# making dataframe
df = pd.read_csv("Bitstamp_BTCUSD_1h.csv", header=1)

#remove columns
df.drop('unix', axis=1, inplace=True)
df.drop('symbol', axis=1, inplace=True)

# convert the 'date' column to datetime format
df['date'] = pd.to_datetime(df["date"])

#set index
df.set_index('date')

#create date features
df['Year'] = df['date'].dt.year
df['Month'] = df['date'].dt.month
df['Week'] = df['date'].dt.isocalendar().week
df['DayOfWeek'] = df['date'].dt.dayofweek
df['Day'] = df['date'].dt.day
df['Hour'] = df['date'].dt.hour

#calc column - if close price is greater than open then bullish otherwise bearish
#df['BullBear'] = np.where(df['close'] > df['open'], 1, 0)


#split data test and train
dateToSplitFrom = max(df['date']) - timedelta(weeks=42)
data_train_filter = df.loc[(df['date'] <= dateToSplitFrom)]
data_train = list(data_train_filter)[1:6]
data_test = df.loc[(df['date'] > dateToSplitFrom)]

print(data_train.columns)
#MinMaxScaler
scaler = MinMaxScaler()

#scaler.fit(data_train)
#scaled_train = scaler.transform(data_train)