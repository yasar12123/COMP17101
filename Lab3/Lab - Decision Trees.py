import pandas as pd
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import graphviz

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

#Decision Trees
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)
tree.plot_tree(clf)

#plot graph
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("forest")

