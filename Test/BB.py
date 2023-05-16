import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

from matplotlib import pyplot as plt


#Read the csv file
df = pd.read_csv("BTC-USD.csv")
#datetime col
df['datetime'] = pd.to_datetime(df["Date"], dayfirst=True)
#set index
df.set_index('datetime', inplace=True)
#sorted df by datetime
df = df.sort_values(by='datetime', ascending=True)

#calc column - if close price is greater than open then bullish otherwise bearish
df['BullBear'] = np.where(df['Close'] > df['Open'], 1, 0)

#remove columns
df.drop('Adj Close', axis=1, inplace=True)

# pd.set_option("display.max.columns", None)
# print(df)
# df.plot.line(y='Close', x='datetime')
# plt.show()

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = df.iloc[:-100]
test = df.iloc[-100:]

predictors = list(df)[1:6]
model.fit(train[predictors], train["BullBear"])

# preds = model.predict(test[predictors])
# preds = pd.Series(preds, index=test.index)
# predScore = precision_score(test["BullBear"], preds)
# print(predScore)
# combined = pd.concat([test["BullBear"], preds], axis=1)
# combined.plot()
# plt.show()

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["BullBear"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["BullBear"], preds], axis=1)
    return combined


def backtest(data, model, predictors, start=365, step=30):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)


predictions = backtest(df, model, predictors)
predictions["Predictions"].value_counts()
ps = precision_score(predictions["BullBear"], predictions["Predictions"])

print(ps)