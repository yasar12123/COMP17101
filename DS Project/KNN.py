from SplitFunction import Dataset

from sklearn.neighbors import KNeighborsClassifier


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


#Read the csv file
dfRaw = pd.read_csv("Gemini_BTCUSD_1h.csv", header=1)

#remove columns
dfRaw.drop('unix', axis=1, inplace=True)
dfRaw.drop('symbol', axis=1, inplace=True)

# convert the 'date' column to datetime format
dfRaw['datetime'] = pd.to_datetime(dfRaw["date"])

#sort by date
df_sorted = dfRaw.sort_values(by=["date"], ascending=True)
#reset index
df = df_sorted.reset_index(drop=True)

#create date features
df['Year'] = df['datetime'].dt.year
df['Month'] = df['datetime'].dt.month
df['Week'] = df['datetime'].dt.isocalendar().week
df['DayOfWeek'] = df['datetime'].dt.dayofweek
df['Day'] = df['datetime'].dt.day
df['Hour'] = df['datetime'].dt.hour
#misc
df['prevClose'] = df['close'].shift(1)
df['Log Return'] = np.log(df['close']/df['open'])
df['BullishBearish'] = df['Log Return'].apply(lambda x: 1 if x > 0 else 0)
#print(df.columns)

a = Dataset(df, ['open', 'high', 'low'], ['BullishBearish'] )
xtrain, ytrain, xtest, ytest = a.x_y_train_test_split(0.8)


print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)

clf = KNeighborsClassifier(7)  # Model the output based on 7 "nearest" examples
clf.fit(xtrain, ytrain)

y_pred = clf.predict(xtest)

_ = pd.DataFrame({'y_true': ytest, 'y_pred': y_pred}).plot(figsize=(15, 2), alpha=.7)
print('Classification accuracy: ', np.mean(ytest == y_pred))
plt.show()

df2 = a.test_dataset_with_prediction(y_pred)

print(df2[['date','open','close','BullishBearish','predicted value']])

