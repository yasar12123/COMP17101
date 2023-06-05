from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense

from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef
from sklearn.metrics import confusion_matrix
import seaborn as sns
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
results = [3, 2, 1]
#create feature
dfSorted['BullishBearish'] = np.select(conditions, results)

#TA indicators
dfSorted['RSI14'] = ta.rsi(dfSorted['close'], 14)
dfSorted['EMA50'] = ta.ema(dfSorted['close'], 50)

#drop nan values
df = dfSorted.dropna()


#paramters for train test window split
dataframe = df
features = ['open', 'Week', 'DayOfWeek', 'RSI14', 'EMA50']
target = ['BullishBearish']
split_ratio = 0.8  # percentage for training
n_future = 1  # Number of days we want to look into the future based on the past days.
n_past = 14   # Number of past days we want to use to predict the future.

#train df split
split = int(len(dataframe) * split_ratio)
train_split = dataframe[:split]
train_X_df = train_split[features]
train_Y_df = train_split[target]
#test df split
test_split = dataframe[split:]
dfTestSplit = test_split
test_X_df = test_split[features]
test_Y_df = test_split[target]

#scale data using min max scaler
scalerF = MinMaxScaler()
scalerX = scalerF.fit(train_X_df)
train_X_scaled = scalerX.transform(train_X_df)
test_X_scaled = scalerX.transform(test_X_df)

#one hot encoder
train_Y_encoded = to_categorical(train_Y_df)
test_Y_encoded = to_categorical(test_Y_df)

#rearrange train data into sliding window
trainX = []
trainY = []
for i in range(n_past, len(train_X_scaled) - n_future + 1):
    trainX.append(train_X_scaled[i - n_past:i])
    trainY.append(train_Y_encoded[i + n_future - 1])

#rearrange test data into sliding window
testX = []
testY = []
for i in range(n_past, len(test_X_scaled) - n_future + 1):
    testX.append(test_X_scaled[i - n_past:i])
    testY.append(test_Y_encoded[i + n_future - 1])

#convert to numpy array
xtrain = np.array(trainX)
ytrain = np.array(trainY)
xtest = np.array(testX)
ytest = np.array(testY)


#train test shapes
print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)


# Model
model = Sequential()
model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=(xtrain.shape[1], xtrain.shape[2])))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(ytrain.shape[1], activation='softmax'))  # 3 classes: 3 - Bullish, 2 - Bearish, 1 - Neutral
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(xtrain, ytrain, epochs=3, batch_size=32, validation_split=0.1)

#plt training validation
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

#make predictions and reverse to_categorical
y_predictions = np.argmax(model.predict(xtest), axis=-1)
print(y_predictions)


#reverse to_categorical for yest
rev_cat_ytest = np.argmax(ytest, axis=-1)


# Calculate classification metrics
micro = precision_recall_fscore_support(rev_cat_ytest, y_predictions, average="micro")
macro = precision_recall_fscore_support(rev_cat_ytest, y_predictions, average="macro")
mcc = matthews_corrcoef(rev_cat_ytest, y_predictions)
scores = pd.DataFrame(columns=['name','precision (micro)', 'recall (micro)', 'fscore (micro)', 'support1',
                               'precision (macro)', 'recall (macro)', 'fscore (macro)', 'support2', 'mcc'])
scores.loc[len(scores)] = ['BI-LSTM', micro[0], micro[1], micro[2], micro[3],
                            macro[0], macro[1], macro[2], macro[3], mcc]

pd.set_option("display.max.columns", None)
print(scores)


#plot confusion matrix
cm = confusion_matrix(rev_cat_ytest,  y_predictions)
cm_df = pd.DataFrame(cm, index=[3, 2, 1], columns=['3', '2', '1'])
# Plotting the confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix - BI-LSTM')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()



# a = dataset.actual_predicted_target_values_classification(prediction)
# print(a[["Date", "BullishBearish", "predicted value"]])

