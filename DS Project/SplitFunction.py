from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef

class Dataset(object):
    '''
    '''

    def __init__(self, dataframe, features, target):
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.dfTrainSplit = []
        self.dfTestSplit = []
        self.scalerFeatures = MinMaxScaler()
        self.scalerTarget = MinMaxScaler()
        self.trainX = []
        self.trainY = []
        self.testX = []
        self.testY = []
        self.scalerX = 0
        self.scalerY = 0

    def x_y_train_test_split(self, split_ratio):
        #split data
        dataframe = self.dataframe
        split = int(len(dataframe) * split_ratio)
        #train split
        train_split = dataframe[:split]
        train_X = train_split[self.features]
        train_Y = train_split[self.target]

        # test split
        test_split = dataframe[split:]
        test_X = test_split[self.features]
        test_Y = test_split[self.target]
        self.dfTestSplit = test_split

        #scalers
        scalerF = MinMaxScaler(feature_range=(-1, 1))
        scalerX = scalerF.fit(train_X)
        scalerT = MinMaxScaler(feature_range=(-1, 1))
        scalerY = scalerT.fit(train_Y)
        self.scalerFeatures = scalerF
        self.scalerTarget = scalerT
        self.scalerX = scalerX
        self.scalerY = scalerY


        #scale trainX testX
        train_X_scaled = scalerX.transform(train_X)
        test_X_scaled = scalerX.transform(test_X)
        # scale trainY testY
        train_Y_scaled = scalerY.transform(train_Y)
        test_Y_scaled = scalerY.transform(test_Y)

        train_Y_scaled_final = []
        for x in train_Y_scaled:
            train_Y_scaled_final.append(x[0])

        test_Y_scaled_final = []
        for x in test_Y_scaled:
            test_Y_scaled_final.append(x[0])

        return train_X_scaled, np.array(train_Y_scaled_final), test_X_scaled, np.array(test_Y_scaled_final)

    def test_dataset_with_prediction(self, predictY):
        scalerY = self.scalerY
        predictY_list = scalerY.inverse_transform(predictY.reshape(-1, 1))
        predictY_values = []
        for x in predictY_list:
            predictY_values.append(x[0])
        df = self.dfTestSplit[-len(predictY):].copy()
        df['predicted value'] = predictY_values

        return df

    def classification_models(self, clf_names, classifiers, xtrain, ytrain, xtest, ytest):
        scores = pd.DataFrame(columns=['name', 'precision (micro)', 'recall (micro)', 'fscore (micro)', 'support1'
                                             , 'precision (micro)', 'recall (micro)', 'fscore (micro)', 'support2'
                                             , 'mcc'])
        for name, clf in zip(clf_names, classifiers):
            print("fitting classifier", name)
            clf.fit(xtrain, ytrain)
            print("predicting labels for classifier", name)
            Y_pred = clf.predict(xtest)
            micro = precision_recall_fscore_support(ytest, Y_pred, average="micro")
            macro = precision_recall_fscore_support(ytest, Y_pred, average="macro")
            mcc = matthews_corrcoef(ytest, Y_pred)
            scores.loc[len(scores)] = [name, micro[0], micro[1], micro[2], micro[3],
                                             macro[0], macro[1], macro[2], macro[3],
                                             mcc]
        return scores


