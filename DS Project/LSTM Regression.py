from ClassTrainTestWindowSplit import Dataset

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error
from math import sqrt

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


#Read the csv file
df = pd.read_csv("BTC-USD.csv")

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
#log return
df['Log Return'] = np.log(df['Close']/df['Open'])
df['BullishBearish'] = df['Log Return'].apply(lambda x: 1 if x > 0 else 0)
#df = df[0:100]

#split data sliding window
a = Dataset(df, ['datetime'], ['Open', 'Week', 'DayOfWeek', 'Day'], ['Log Return'])
xtrain, ytrain, xtest, ytest = a.sliding_window_split(0.8, 50, 2)


print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)


#LSTM Model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(xtrain.shape[1], xtrain.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(ytrain.shape[1]))
model.compile(optimizer='adam', loss='mse', metrics=["accuracy"])
model.summary()


# fit the model
history = model.fit(xtrain, ytrain, epochs=30, batch_size=32, validation_split=0.1, verbose=1)
#plt training validation
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()


#make predictions
prediction = model.predict(xtest)

a = a.actual_predicted_target_values(prediction)
print(a[["Date", "Log Return", "predicted value"]])

# calculate RMSE
rmse = sqrt(mean_squared_error(a[['Log Return']], a[['predicted value']]))
print('Test RMSE: %.3f' % rmse)

# line plot for actual and predictions
ax = plt.gca()
a.plot(kind='line',
        x='Date',
        y='Log Return',
        color='green',ax=ax)
a.plot(kind='line',
        x='Date',
        y='predicted value',
        color='orange',ax=ax)
plt.legend()
plt.show()
