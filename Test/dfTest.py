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

#df = df[0:10]
dataFeatures = df[['Date', 'Open', 'High', 'Low']]
dataTarget = df[['Close']]


##initialse data into split
slidingWindow = TimeBasedCV(train_period=14, test_period=1, freq='days')
slidingWindow.split(df)
x_y_train_test_split = slidingWindow.x_y_split(0.8, dataFeatures, dataTarget)


#x y train test data
x_train = x_y_train_test_split[0]
y_train = x_y_train_test_split[1]
x_test = x_y_train_test_split[2]
y_test = x_y_train_test_split[3]

#print(x_train)
#print(y_train)
print(x_test[-1])
print(y_test[-1])
