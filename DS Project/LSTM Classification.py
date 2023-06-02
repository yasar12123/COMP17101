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
xtrain, ytrain, xtest, ytest = dataset.sliding_window_split(0.8, 14, 1)

#train test shapes
print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)



#LSTM Model
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(xtrain.shape[1], xtrain.shape[2])))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(5))
model.add(Activation('softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='RMSprop')
model.summary()
# fit the model
history = model.fit(xtrain, ytrain, epochs=4, batch_size=32, validation_split=0.1, verbose=1, shuffle=False)





#make predictions
prediction = model.predict(xtest)
dfWithPred = dataset.actual_predicted_target_values(prediction)
print(dfWithPred[["Date", "BullishBearish", "predicted value"]])



# #score
# micro = precision_recall_fscore_support(ytest_inv, Y_pred_inv, average="micro")
# macro = precision_recall_fscore_support(ytest_inv, Y_pred_inv, average="macro")
# mcc = matthews_corrcoef(ytest_inv, Y_pred_inv)



# Plot Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy Model')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Plot Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Model')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()