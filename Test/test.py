import numpy as np
from sklearn.model_selection import TimeSeriesSplit
X = np.array([['a', 'b'], ['c', 'd'], ['e', 'f'], ['g', 'h'], ['i', 'j'], ['k', 'l']])
y = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
time_series = TimeSeriesSplit()
print(time_series)
# for train_index, test_index in time_series.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

n_future = 24   # Number of days we want to look into the future based on the past days.
n_past = 336  # Number of past days we want to use to predict the future.


for train_index, test_index in time_series.split(X):
    print("TRAIN:", train_index,  "TEST:", test_index)
    print(X[train_index], X[test_index], y[train_index], y[test_index])
    #trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    #trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])