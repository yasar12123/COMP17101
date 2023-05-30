from preProcessData import df
from SplitFunction import Dataset
from matplotlib import pyplot as plt
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn import metrics

import seaborn as sns
from matplotlib import pyplot as plt


a = Dataset(df, ['open', 'high', 'low'], ['BullishBearish'] )
xtrain, ytrain, xtest, ytest = a.x_y_train_test_split(0.8)


# Fit regression model
regr = DecisionTreeRegressor(max_depth=2)
regr.fit(xtrain, ytrain)
y_pred = regr.predict(xtest)

print(y_pred)


# sns.lineplot(x='date', y='rsi', data=df)
# sns.lineplot(df['rsiAbove70'])
# sns.lineplot(df['Log Return'])
# sns.lineplot(df['close'])
# plt.show()
#


