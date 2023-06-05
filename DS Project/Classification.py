from preProcessData import dfDaily
from ClassMachineLearning import dataset_features_target

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import math

from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


#split date into x y
dataset = dataset_features_target(dfDaily, ['RSI14', 'EMA14', 'PrevPercentChange'], ['BullishBearish'] )
xtrain, ytrain, xtest, ytest = dataset.x_y_train_test_split(0.8)


# #to get best knn
# #method 1 sqrt of n
# k = math.sqrt(len(xtrain))
# #method 2 error rate
# error_rate = []
# for i in range(1,50):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(xtrain, ytrain)
#     pred = knn.predict(xtest)
#     error_rate.append(np.mean(pred != ytest))
#
# plt.figure(figsize=(15,10))
# plt.plot(range(1,50),error_rate, marker='o', markersize=9)
# plt.show()




#classification models
classifiers = [KNeighborsClassifier(4),
               KNeighborsClassifier(12),
               KNeighborsClassifier(47),
               GaussianNB(),
               DecisionTreeClassifier(max_depth=7, random_state=123),
               DecisionTreeClassifier(max_depth=60, random_state=123),
               RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=123),
               RandomForestClassifier(n_estimators=1000, max_depth=20, random_state=123),
               MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=10000, activation='relu', random_state=123),
               LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=123),
               LinearSVC(multi_class='ovr', class_weight='balanced', random_state=123) ]

clf_names = ["Nearest Neighbors (k=4)",
             "Nearest Neighbors (k=12)",
             "Nearest Neighbors (k=47)",
             "GaussianNB",
             "Decision Tree (Max Depth=7)",
             "Decision Tree (Max Depth=60)",
             "Random Forest (Max Depth=5)",
             "Random Forest (Max Depth=20)",
             "MLP (RelU)",
             "Logistic Regression",
             "LinearSVC"]


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



# #plot confusion matrix
for x in predictions:
    cm = confusion_matrix(ytest,  x[1])
    cm_df = pd.DataFrame(cm,
                         index=[3, 2, 1],
                         columns=['3', '2', '1'])
    # Plotting the confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True)
    plt.title('Confusion Matrix - ' + x[0])
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values')
    plt.show()
