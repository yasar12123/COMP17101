from TimeBasedCV import TimeBasedCV
import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


#Read the csv file
df = pd.read_csv("BTC-USD.csv")
#pre processing
#datetime col
df['datetime'] = pd.to_datetime(df["Date"], dayfirst=True)

df = df[0:10]
dataFeatures = df[['Date', 'Open', 'High', 'Low']]
dataTarget = df[['Close']]


#get index of data
tscv = TimeBasedCV(train_period=3, test_period=1, freq='days')
index_output = (tscv.split(df))

#train and test split index
split = int(tscv.n_splits * 0.8)
trainSplit = index_output[:split]
testSplit = index_output[split:]


#organise data into x y train
x_train = []
y_train = []
for loopIndex, trainIndex in enumerate(trainSplit):
    x_trainIndexMin = min(trainIndex[0])
    x_trainIndexMax = max(trainIndex[0]) + 1
    y_trainIndex = trainIndex[1][0]
    x_trainDF = dataFeatures[x_trainIndexMin:x_trainIndexMax]
    y_trainDF = dataTarget[y_trainIndex:y_trainIndex+1]
    x_train.append(x_trainDF.to_numpy())
    y_train.append(y_trainDF.values.tolist()[0][0])

# print(x_train[0])
# print('-----')
# print(y_train)


# organise data into x y test
x_test = []
y_test = []
for loopIndex, testIndex in enumerate(testSplit):
    x_testIndexMin = min(testIndex[0])
    x_testIndexMax = max(testIndex[0]) + 1
    y_testIndex = testIndex[1][0]
    x_testDF = dataFeatures[x_testIndexMin:x_testIndexMax]
    y_testDF = dataTarget[y_testIndex:y_testIndex + 1]
    x_test.append(x_testDF.to_numpy())
    y_test.append(y_testDF.values.tolist()[0][0])

print(x_test)
