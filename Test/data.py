import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from datetime import datetime, date, timedelta

#Read the csv file
df = pd.read_csv("Bitstamp_BTCUSD_1h.csv", header=1)

#remove columns
df.drop('unix', axis=1, inplace=True)
df.drop('symbol', axis=1, inplace=True)

# convert the 'date' column to datetime format
df['date'] = pd.to_datetime(df["date"])

#create date features
df['Year'] = df['date'].dt.year
df['Month'] = df['date'].dt.month
df['Week'] = df['date'].dt.isocalendar().week
df['DayOfWeek'] = df['date'].dt.dayofweek
df['Day'] = df['date'].dt.day
df['Hour'] = df['date'].dt.hour

#Separate dates for future plotting
train_dates = pd.to_datetime(df['date'])

#Variables for training
cols = list(df)[1:]
df_for_training = df[cols].astype(float)

#scaling
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)


#Empty lists to be populated using formatted training data
trainX = []
trainY = []

n_future = 24   # Number of days we want to look into the future based on the past days.
n_past = 336  # Number of past days we want to use to predict the future.

#Reformat input data into a shape: (n_samples x timesteps x n_features)
#In my example, my df_for_training_scaled has a shape (12823, 5)
#12823 refers to the number of data points and 5 refers to the columns (multi-variables).
for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

#convert to array
trainX, trainY = np.array(trainX), np.array(trainY)

# print('trainX shape == {}.'.format(trainX.shape))
# print('trainY shape == {}.'.format(trainY.shape))


# model = Sequential()
# model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
# model.add(LSTM(32, activation='relu', return_sequences=False))
# model.add(Dropout(0.2))
# model.add(Dense(trainY.shape[1]))
#
# model.compile(optimizer='adam', loss='mse')
# model.summary()
#
# # fit the model
# history = model.fit(trainX, trainY, epochs=5, batch_size=16, validation_split=0.1, verbose=1)
#
# plt.plot(history.history['loss'], label='Training loss')
# plt.plot(history.history['val_loss'], label='Validation loss')
# plt.legend()
# plt.show()
