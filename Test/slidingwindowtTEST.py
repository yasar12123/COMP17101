from ClassSlidingWindow import SlidingWindow
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd

#Read the csv file
df = pd.read_csv("BTC-USD.csv")
#pre processing
#datetime col
df['datetime'] = pd.to_datetime(df["Date"], dayfirst=True)
df = df[0:100]
dataFeatures = df[['Open', 'High', 'Low']]
dataTarget = df[['Close']]


a = SlidingWindow(df, ['Open', 'High', 'Low'], ['Close'])

x,y = a.split()
#scaler = a.scaler
#print(y)
inx, iny = a.inverse_scaler()
#print(iny)


print(y.shape)



