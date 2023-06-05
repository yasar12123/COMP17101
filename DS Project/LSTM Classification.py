from preProcessData import dfDaily
from sklearn.preprocessing import MinMaxScaler
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dense, Dropout

from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta



#paramters for train test window split
dataframe = dfDaily
features = ['close', 'PrevPercentChange', 'Volume USD', 'Volume BTC', 'RSI14', 'EMA14', 'STOCHk_14_3_3', 'STOCHd_14_3_3']
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
scalerF = MinMaxScaler(feature_range=(-1,1))
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
model.add(Bidirectional(LSTM(64, activation='relu', return_sequences=True), input_shape=(xtrain.shape[1], xtrain.shape[2])))
model.add(Bidirectional(LSTM(32, activation='relu')))
model.add(Dropout(0.1))
model.add(Dense(ytrain.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(xtrain, ytrain, epochs=20, batch_size=32, validation_split=0.1)

#plt training validation
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

#make predictions and reverse to_categorical labels
pred = model.predict(xtest)
y_predictions = np.argmax(pred, axis=-1)
#reverse to_categorical for ytest
rev_cat_ytest = np.argmax(ytest, axis=-1)


# Calculate classification metric scores
micro = precision_recall_fscore_support(rev_cat_ytest, y_predictions, average="micro")
macro = precision_recall_fscore_support(rev_cat_ytest, y_predictions, average="macro")
mcc = matthews_corrcoef(rev_cat_ytest, y_predictions)
scores = pd.DataFrame(columns=['name','precision (micro)', 'recall (micro)', 'fscore (micro)', 'support1',
                               'precision (macro)', 'recall (macro)', 'fscore (macro)', 'support2', 'mcc'])
scores.loc[len(scores)] = ['BI-LSTM', micro[0], micro[1], micro[2], micro[3],
                            macro[0], macro[1], macro[2], macro[3], mcc]

pd.set_option("display.max.columns", None)
print(scores)

# Create bar plot for scores
ax = plt.gca()
scores.plot(kind='barh', x='name', y=scores.columns[1:], ax=ax, figsize=(20,10))
for container in ax.containers:
    ax.bar_label(container)
plt.legend()
plt.show()

#plot confusion matrix
cm = confusion_matrix(rev_cat_ytest,  y_predictions)
cm_df = pd.DataFrame(cm, index=[3, 2, 1], columns=['Bullish', 'Bearish', 'Neutral'])
# Plotting the confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix - BI-LSTM')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()

