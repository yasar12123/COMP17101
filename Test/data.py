import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support


# making dataframe
df = pd.read_csv("BTC-USD.csv")
#print(df.head())

#split data and target
x = df.drop('Close', axis=1)
y = df.Close

#train test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)


#k neighbor
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(x_train, y_train)
#print(neigh.score(x_train, y_train))
#predictTarget = neigh.predict(x_test)
#print(predictTarget)
#print(precision_recall_fscore_support(y_test, predictTarget, average='macro'))