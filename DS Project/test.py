from ClassTrainTestWindowSplit import Dataset

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Dense, Dropout, Activation
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef
from keras.layers import Dense, Dropout, Flatten, Reshape
import keras
import tensorflow as tf


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta


#Read the csv file
dfRaw = pd.read_csv("Gemini_BTCUSD_1h.csv", header=1)

# convert the 'date' column to datetime format
dfRaw['datetime'] = pd.to_datetime(dfRaw["date"])

#sort by date
dfSorted = dfRaw.sort_values(by=["datetime"], ascending=True)
#reset index
dfSorted = dfSorted.reset_index(drop=True)
#set datetime as index
#dfSorted = dfSorted.set_index('datetime')

#remove columns
dfSorted.drop('unix', axis=1, inplace=True)
dfSorted.drop('symbol', axis=1, inplace=True)
dfSorted.drop('date', axis=1, inplace=True)

#create date/time features
dfSorted['Date'] = dfSorted['datetime'].dt.date
dfSorted['Hour'] = dfSorted['datetime'].dt.hour
dfSorted['Year'] = dfSorted['datetime'].dt.year
dfSorted['Month'] = dfSorted['datetime'].dt.month
dfSorted['Week'] = dfSorted['datetime'].dt.isocalendar().week
dfSorted['DayOfWeek'] = dfSorted['datetime'].dt.dayofweek
dfSorted['DayOfMonth'] = dfSorted['datetime'].dt.day

#next day close price
dfSorted['NextClose'] = dfSorted['close'].shift(-1)
#percetange increase/decrease between close price and nextClose price
dfSorted['PercentChange'] = ((dfSorted['NextClose'] - dfSorted['close']) / dfSorted['close']) * 100

#define conditions
conditions = [dfSorted['PercentChange'] >= 0.25,
              dfSorted['PercentChange'] <= -0.25,
             (dfSorted['PercentChange'] > -0.25) & (dfSorted['PercentChange'] < 0.25)]
#define results
results = [1, -1, 0]
#create feature
dfSorted['BullishBearish'] = np.select(conditions, results)

#TA indicators
dfSorted['RSI14'] = ta.rsi(dfSorted['close'], 14)
dfSorted['EMA50'] = ta.ema(dfSorted['close'], 50)

#drop nan values
df = dfSorted.dropna()



#split data sliding window
dataset = Dataset(df, ['datetime'], ['open', 'Week', 'DayOfWeek', 'RSI14', 'EMA50'], ['BullishBearish'])
xtrain, ytrain, xtest, ytest = dataset.sliding_window_split_classification(0.8, 14, 1)

#train test shapes
print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)

verbose, epochs, batch_size = 0, 15, 32
n_timesteps, n_features, n_outputs = xtrain.shape[1], xtrain.shape[2], ytrain.shape[1]
model = Sequential()
model.add(LSTM(64, input_shape=(n_timesteps,n_features), return_sequences=True))
model.add(LSTM(32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# fit network
history = model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size, verbose=verbose)

#plt training validation
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()



