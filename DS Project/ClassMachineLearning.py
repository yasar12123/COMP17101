from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef

class dataset_features_target(object):
    '''
    This class can be used to fee in the dataframe, with features and target values, it can then split
    the x,y, train,test, scale the data using MinMaxScaler (feature range -1, 1). This can then further
    train models with the data, create predictions and then compares the results.
    '''

    def __init__(self, dataframe, features, target):
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.dfTrainSplit = []
        self.dfTestSplit = []
        self.scalerFeatures = MinMaxScaler(feature_range=(-1, 1))
        self.scalerTarget = MinMaxScaler(feature_range=(-1, 1))
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
        scalerF = self.scalerFeatures
        scalerX = scalerF.fit(train_X)
        self.scalerX = scalerX
        scalerT = self.scalerTarget
        scalerY = scalerT.fit(train_Y)
        self.scalerY = scalerY

        # scale trainX testX
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

    def inverse_y_scaler(self, y):
        scalerY = self.scalerY
        Y_list = scalerY.inverse_transform(y.reshape(-1, 1))
        Y_values = []
        for x in Y_list:
            Y_values.append(x[0])
        return np.array(Y_values)

    def test_dataset_with_prediction(self, name, predictY):
        scalerY = self.scalerY
        predictY_list = scalerY.inverse_transform(predictY.reshape(-1, 1))
        predictY_values = []
        for x in predictY_list:
            predictY_values.append(x[0])
        df = self.dfTestSplit[-len(predictY):].copy()
        df[name+' PredictedValue'] = predictY_values

        return df

    def classification_models(self, clf_names, classifiers):
        xtrain, ytrain, xtest, ytest = self.x_train, self.y_train, self.x_test, self.y_test
        dfFinal = self.dfTestSplit[-len(ytest):].copy()
        predictions = []
        scores = pd.DataFrame(columns=['name', 'precision (micro)', 'recall (micro)', 'fscore (micro)', 'support (micro)'
                                             , 'precision (macro)', 'recall (macro)', 'fscore (macro)', 'support (macro)'
                                             , 'mcc'])
        for name, clf in zip(clf_names, classifiers):
            #fit model get pred
            print(f'running model: {name}')
            clf.fit(xtrain, ytrain)
            Y_pred = clf.predict(xtest)
            #inverse test and predict y
            Y_pred_inv = self.inverse_y_scaler(Y_pred)
            ytest_inv = self.inverse_y_scaler(ytest)
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
            dfFinal = pd.merge(dfFinal, dfPred[['Date', (name+' PredictedValue')]], on='Date', how='left')

        return scores, dfFinal, predictions


