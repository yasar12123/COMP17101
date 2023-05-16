import numpy as np
import pandas as pd
from TimeBasedCV import TimeBasedCV
import datetime
from sklearn.preprocessing import StandardScaler


from matplotlib import pyplot as plt


#Read the csv file
df = pd.read_csv("BTC-USD.csv")
#datetime col
#df['datetime'] = pd.to_datetime(df["Date"], dayfirst=True)

df = df[0:10]


#data array
trainDataArray = df.loc[:, df.columns != 'Date'].values
targetDataArray = df.loc[:, df.columns == 'Close'].values


#scale data
#scaler train data
scaler = StandardScaler()
scaler = scaler.fit(trainDataArray)
trainDataScaled = scaler.transform(trainDataArray)
#scaler target data
scaler = scaler.fit(targetDataArray)
targetDataScaled = scaler.transform(targetDataArray)


#split train test data
split = int(len(trainDataScaled) * 0.7)
#x,y arrays before sliding window split
x_train = trainDataScaled[: split]
x_test = trainDataScaled[split: len(trainDataScaled)]
y_train = targetDataScaled[: split]
y_test = targetDataScaled[split: len(targetDataScaled)]


#X,Y array - sliding window split
nSplit = 3
Xtrain = []
Xtest = []
Ytrain = []
Ytest = []
for i in range(nSplit, len(x_train)):
    Xtrain.append(x_train[i-nSplit:i, :x_train.shape[1]])
    Ytrain.append(y_train[i])
for i in range(nSplit, len(x_test)):
    Xtrain.append(x_test[i-nSplit:i, :x_test.shape[1]])
    Ytrain.append(y_test[i])


Xtrain, Ytrain = (np.array(Xtrain), np.array(Ytrain))
Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2]))

# Xtest, Ytest = (np.array(Xtest), np.array(Ytest))
# Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], Xtest.shape[2]))

print(Xtrain.shape, Ytrain.shape)
print(Xtrain.shape, Ytrain.shape)

