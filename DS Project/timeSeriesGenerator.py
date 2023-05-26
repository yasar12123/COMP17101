
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split



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

#create date features
df['Year'] = df['datetime'].dt.year
df['Month'] = df['datetime'].dt.month
df['Week'] = df['datetime'].dt.isocalendar().week
df['DayOfWeek'] = df['datetime'].dt.dayofweek
df['Day'] = df['datetime'].dt.day
df['Hour'] = df['datetime'].dt.hour
#misc
df['prevClose'] = df['close'].shift(1)
df['Log Return'] = np.log(df['close']/df['open'])
df['BullishBearish'] = df['Log Return'].apply(lambda x: 1 if x > 0 else 0)

df2 = df.dropna()

features = df[['open', 'high', 'low']].to_numpy().tolist()
target = df['BullishBearish'].tolist()

#split data into train and test
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=123, shuffle=False)


#split into window series
trainSplit = TimeseriesGenerator(x_train, y_train, length=7, sampling_rate=1, batch_size=1)
testSplit = TimeseriesGenerator(x_test, y_test, length=7, sampling_rate=1, batch_size=1)

print(trainSplit)

