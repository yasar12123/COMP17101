from ClassTrainTestWindowSplit import Dataset

import numpy as np
import pandas as pd


#Read the csv file
df = pd.read_csv("../DS Project/BTC-USD.csv")

#pre processing
#datetime col
df['datetime'] = pd.to_datetime(df["Date"], dayfirst=True)
df['row_number'] = df.reset_index().index
#create date features
df['Year'] = df['datetime'].dt.year
df['Month'] = df['datetime'].dt.month
df['Week'] = df['datetime'].dt.isocalendar().week
df['DayOfWeek'] = df['datetime'].dt.dayofweek
df['Day'] = df['datetime'].dt.day
#df = df[0:100]
df['Log Return'] = np.log(df['Close']/df['Open'])
df['BullishBearish'] = df['Log Return'].apply(lambda x: 1 if x > 0 else 0)

#print(df[['Date','Open', 'Close', 'Log Return','BullishBearish']])

#split data sliding window
a = Dataset(df, ['datetime'], ['Open', 'High', 'Low'], ['Close'])
xtrain, ytrain, xtest, ytest = a.SlidingWindowSplit(0.8, 14, 1)