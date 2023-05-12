import pandas as pd
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support


#import dataset
forest = fetch_covtype()

#convert dataset into pandas dataframe
df = pd.DataFrame(data=np.c_[forest['data'], forest['target']],
                  columns=forest['feature_names'] + ['target'])

#pd.set_option("display.max.columns", None)
#print(df)

#split data train and test
x, y = fetch_covtype(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.95, random_state=42, stratify=y)

#k neighbor
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)
print(neigh.score(x_train, y_train))
predictTarget = neigh.predict(x_test)
print(predictTarget)
print(precision_recall_fscore_support(y_test, predictTarget, average='macro'))
