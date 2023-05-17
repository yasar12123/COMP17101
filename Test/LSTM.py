from TimeBasedCV import TimeBasedCV
import pandas as pd
import matplotlib.pyplot as plt # this is used for the plot the graph
from sklearn.preprocessing import MinMaxScaler

## for Deep-learing:
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout

#Read the csv file
df = pd.read_csv("BTC-USD.csv")
#pre processing
#datetime col
df['datetime'] = pd.to_datetime(df["Date"], dayfirst=True)
dataFeatures = df[['Open', 'High', 'Low']]
dataTarget = df[['Close']]

df = df[0:20]


##initialse data into split
slidingWindow = TimeBasedCV(train_period=3, test_period=1, freq='days')
slidingWindow.split(df)
x_y_train_test_split = slidingWindow.x_y_split(0.8, dataFeatures, dataTarget)



#x y train test data
x_train = x_y_train_test_split[0]
y_train = x_y_train_test_split[1]
x_test = x_y_train_test_split[2]
y_test = x_y_train_test_split[3]

print(y_train[1])
print(slidingWindow.scaler_transform(y_train)[2])

# #lstm model
# model = Sequential()
# model.add(LSTM(100, input_shape=(x_train.shape[1], x_train.shape[2])))
# model.add(Dropout(0.2))
# #    model.add(LSTM(70))
# #    model.add(Dropout(0.3))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
#
# # fit network
# history = model.fit(x_train, y_train, epochs=20, batch_size=70, validation_data=(x_test, y_test), verbose=2, shuffle=False)
#
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
# plt.show()
