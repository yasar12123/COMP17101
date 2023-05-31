from preProcessData import dfDaily
from SplitFunction import Dataset
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn import metrics

import seaborn as sns
from matplotlib import pyplot as plt


a = Dataset(dfDaily, ['RSI14', 'EMA50', 'open'], ['BullishBearish'] )
xtrain, ytrain, xtest, ytest = a.x_y_train_test_split(0.8)



from sklearn.neighbors import KNeighborsClassifier

classifiers = [KNeighborsClassifier(1),
               KNeighborsClassifier(2),
               KNeighborsClassifier(3)]

clf_names = ["Nearest Neighbors (k=1)",
             "Nearest Neighbors (k=2)",
             "Nearest Neighbors (k=3)"]



pd.set_option("display.max.columns", None)

scores = a.classification_models(clf_names, classifiers, xtrain, ytrain, xtest, ytest)

print(scores)



