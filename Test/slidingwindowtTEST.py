from ClassSlidingWindow import SlidingWindow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout

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
a = SlidingWindow(df, ['row_number', 'Open', 'High', 'Low'], ['Close'])
xtrain, ytrain, xtest, ytest = a.split(0.8, 14, 1)


print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)


model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(xtrain.shape[1], xtrain.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(ytrain.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()

# fit the model
history = model.fit(xtrain, ytrain, epochs=20, batch_size=16, validation_split=0.1, verbose=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()