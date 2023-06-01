from preProcessData import dfDaily
from ClassMachineLearning import dataset_features_target

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


#split date into x y
dataset = dataset_features_target(dfDaily, ['RSI14', 'EMA50', 'open'], ['BullishBearish'] )
xtrain, ytrain, xtest, ytest = dataset.x_y_train_test_split(0.8)


#classification models
classifiers = [KNeighborsClassifier(1),
               KNeighborsClassifier(3),
               DecisionTreeClassifier(max_depth=20),
               DecisionTreeClassifier(max_depth=60),
               RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=1),
               RandomForestClassifier(n_estimators=1000, max_depth=20, random_state=1),
               MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=10000, activation='relu'),
               LogisticRegression(multi_class='multinomial', solver='lbfgs')]

clf_names = ["Nearest Neighbors (k=1)",
             "Nearest Neighbors (k=3)",
             "Decision Tree (Max Depth=20)",
             "Decision Tree (Max Depth=60)",
             "Random Forest (Max Depth=5)",
             "Random Forest (Max Depth=20)",
             "MLP (RelU)",
             "Logistic Regression"]


#get the scores for the models
scores, df, predictions = dataset.classification_models(clf_names, classifiers)
pd.set_option("display.max.columns", None)
#print(df)
#print(scores)
#print(predictions[1])


# Create bar plot for scores
ax = plt.gca()
scores.plot(kind='barh', x='name', y=scores.columns[1:], ax=ax, figsize=(20,10))
plt.legend()
plt.show()

# Create bar plot for fscore
ax = plt.gca()
scores.plot(kind='barh', x='name', y=['fscore (micro)','fscore (macro)'], ax=ax, figsize=(20,10))
plt.legend()
plt.show()


#plot confusion matrix
for x in predictions:
    cm = confusion_matrix(ytest,  x[1])
    cm_df = pd.DataFrame(cm,
                         index=[-1, 0, 1],
                         columns=['-1', '0', '1'])
    # Plotting the confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix - ' + x[0])
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()
