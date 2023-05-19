from ClassSlidingWindow import SlidingWindow

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
#df = df[0:100]

#split data sliding window
a = SlidingWindow(df, ['datetime'], ['row_number', 'Open', 'High', 'Low'], ['Close'])
xtrain, ytrain, xtest, ytest = a.split(0.8, 200, 1)


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
print(a[["Date", "Close", "predicted value"]])


# calculate RMSE
rmse = sqrt(mean_squared_error(a[['Close']], a[['predicted value']]))
print('Test RMSE: %.3f' % rmse)

# line plot for math marks
ax = plt.gca()
a.plot(kind='line',
        x='Date',
        y='Close',
        color='green',ax=ax)
a.plot(kind='line',
        x='Date',
        y='predicted value',
        color='orange',ax=ax)
plt.legend()
plt.show()