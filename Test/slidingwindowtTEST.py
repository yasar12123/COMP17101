from ClassSlidingWindow import SlidingWindow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


#Read the csv file
df = pd.read_csv("BTC-USD.csv")
#pre processing
#datetime col
df['datetime'] = pd.to_datetime(df["Date"], dayfirst=True)
df['row_number'] = df.reset_index().index
#df = df[0:100]

#split data sliding window
a = SlidingWindow(df, ['datetime'], ['row_number', 'Open', 'High', 'Low'], ['Close'])
xtrain, ytrain, xtest, ytest = a.split(0.8, 14, 1)


print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)


b  = a.actual_predicted_target_values(ytest)

print(np.array(b))
#print(b[["Date","Close","predicted value"]])
