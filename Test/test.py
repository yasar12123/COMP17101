import numpy as np
import pandas as pd
from TimeBasedCV import TimeBasedCV
import datetime
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


#Read the csv file
df = pd.read_csv("BTC-USD.csv")
#pre processing
#datetime col
df['datetime'] = pd.to_datetime(df["Date"], dayfirst=True)

df = df[0:20]
dataFeatures = df[['Date', 'Open', 'High', 'Low']]
dataTarget = df[['Close']]



steps = 3
split_ratio = 0.8

for loopIndex, trainIndex in enumerate(dataTarget):
    print(trainIndex)

