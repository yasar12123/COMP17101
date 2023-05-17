from TimeBasedCV import TimeBasedCV
import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

#Read the csv file
df = pd.read_csv("BTC-USD.csv")
#pre processing
#datetime col
df['datetime'] = pd.to_datetime(df["Date"], dayfirst=True)
df = df[0:100]
dataFeatures = df[['Open', 'High', 'Low']].astype(float)
dataTarget = df[['Close']].astype(float)


#get index of data
tscv = TimeBasedCV(train_period=3, test_period=1, freq='days')
index_output = (tscv.split(df))

#train and test split index
split = int(tscv.n_splits * 0.8)
trainSplit = index_output[:split]
testSplit = index_output[split:]



# organise data into x y train
x_train = []
y_train = []
scaler = MinMaxScaler()

for loopIndex, trainIndex in enumerate(trainSplit):
    x_trainIndexMin = min(trainIndex[0])
    x_trainIndexMax = max(trainIndex[0]) + 1
    x_trainDF = dataFeatures[x_trainIndexMin:x_trainIndexMax]
    x_trainDFScaled = scaler.fit_transform(x_trainDF.values)
    x_train.append(x_trainDFScaled)
    y_trainIndexMin = min(trainIndex[1])
    y_trainIndexMax = max(trainIndex[1]) + 1
    y_trainDF = dataTarget[y_trainIndexMin:y_trainIndexMax]
    y_trainDFScaled = scaler.fit_transform(y_trainDF.values)
    y_train.append(y_trainDFScaled[0][0])

# organise data into x y test
x_test = []
y_test = []
for loopIndex, testIndex in enumerate(testSplit):
    x_testIndexMin = min(testIndex[0])
    x_testIndexMax = max(testIndex[0]) + 1
    x_testDF = dataFeatures[x_testIndexMin:x_testIndexMax]
    x_testDFScaled = scaler.fit_transform(x_testDF.values)
    x_test.append(x_testDFScaled)
    y_testIndexMin = min(testIndex[1])
    y_testIndexMax = max(testIndex[1]) + 1
    y_testDF = dataTarget[y_testIndexMin:y_testIndexMax]
    y_testDFScaled = scaler.fit_transform(y_testDF.values)
    y_test.append(y_testDFScaled[0])


