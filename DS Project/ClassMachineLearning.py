from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef

class dataset_features_target(object):
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
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
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

        self.x_train, self.y_train = train_X_scaled, np.array(train_Y_scaled_final)
        self.x_test, self.y_test = test_X_scaled, np.array(test_Y_scaled_final)
        return train_X_scaled, np.array(train_Y_scaled_final), test_X_scaled, np.array(test_Y_scaled_final)

    def test_dataset_with_prediction(self, name, predictY):
        scalerY = self.scalerY
        predictY_list = scalerY.inverse_transform(predictY.reshape(-1, 1))
        predictY_values = []
        for x in predictY_list:
            predictY_values.append(x[0])
        df = self.dfTestSplit[-len(predictY):].copy()
        df[name+'predicted value'] = predictY_values

        return df

    def classification_models(self, clf_names, classifiers):
        xtrain, ytrain, xtest, ytest = self.x_train, self.y_train, self.x_test, self.y_test
        dfFinal = self.dfTestSplit[-len(ytest):].copy()
        predictions = []
        scores = pd.DataFrame(columns=['name', 'precision (micro)', 'recall (micro)', 'fscore (micro)', 'support1'
                                             , 'precision (macro)', 'recall (macro)', 'fscore (macro)', 'support2'
                                             , 'mcc'])
        for name, clf in zip(clf_names, classifiers):
            #fit model get pred
            clf.fit(xtrain, ytrain)
            Y_pred = clf.predict(xtest)
            #inverse test and predict y
            scalerY = self.scalerY
            Y_pred_inv = scalerY.inverse_transform(Y_pred.reshape(-1, 1))
            ytest_inv = scalerY.inverse_transform(ytest.reshape(-1, 1))
            #calc score
            micro = precision_recall_fscore_support(ytest_inv, Y_pred_inv, average="micro")
            macro = precision_recall_fscore_support(ytest_inv, Y_pred_inv, average="macro")
            mcc = matthews_corrcoef(ytest_inv, Y_pred_inv)
            predictions.append([name, Y_pred_inv])
            scores.loc[len(scores)] = [name, micro[0], micro[1], micro[2], micro[3],
                                             macro[0], macro[1], macro[2], macro[3],
                                             mcc]
            #inverse predict y with dataframe
            dfPred = self.test_dataset_with_prediction(name, Y_pred)
            #merge columns to final dataframe
            dfFinal = pd.merge(dfFinal, dfPred[['Date', (name+'predicted value')]], on='Date', how='left')

        return scores, dfFinal, predictions


