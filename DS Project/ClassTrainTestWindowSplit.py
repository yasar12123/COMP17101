from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

class Dataset(object):
    '''
    '''

    def __init__(self, dataframe, datetime_feature, features, target):
        self.dataframe = dataframe
        self.datetime_feature = dataframe[datetime_feature]
        self.features = features
        self.target = target
        self.dfTrainSplit = []
        self.dfTestSplit = []
        self.scalerFeatures = MinMaxScaler(feature_range=(-1, 1))
        self.scalerTarget = MinMaxScaler(feature_range=(-1, 1))
        self.trainX = []
        self.trainY = []
        self.testX = []
        self.testY = []
        self.scalerX = 0
        self.scalerY = 0

    def sliding_window_split(self, split_ratio, steps_back, steps_forward):
        #train split
        split = int(len(self.dataframe) * split_ratio)
        train_split = self.dataframe[:split]
        self.dfTrainSplit = train_split
        train_X = train_split[self.features]
        train_Y = train_split[self.target]

        #scale data
        scalerF = self.scalerFeatures
        # scale trainX
        scalerX = scalerF.fit(train_X)
        train_X_scaled = scalerX.transform(train_X)
        self.scalerX = scalerX
        # scale trainY
        scalerT = self.scalerTarget
        scalerY = scalerT.fit(train_Y)
        train_Y_scaled = scalerY.transform(train_Y)
        self.scalerY = scalerY

        #number of steps to look back and forward
        n_future = steps_forward  # Number of days we want to look into the future based on the past days.
        n_past = steps_back   # Number of past days we want to use to predict the future.

        #split train data into steps
        trainX = []
        trainY = []
        for i in range(n_past, len(train_X_scaled) - n_future + 1):
            trainX.append(train_X_scaled[i - n_past:i])
            trainY.append(train_Y_scaled[i + n_future - 1])

        self.trainX, self.trainY = np.array(trainX), np.array(trainY)

        #test split
        test_split = self.dataframe[split:]
        self.dfTestSplit = test_split
        test_X = test_split[self.features]
        test_Y = test_split[self.target]

        #scale data
        # scale testX
        test_X_scaled = scalerX.transform(test_X)
        # scale trainY
        test_Y_scaled = scalerY.transform(test_Y)

        #split test data into steps
        testX = []
        testY = []
        for i in range(n_past, len(test_X_scaled) - n_future + 1):
            testX.append(test_X_scaled[i - n_past:i])
            testY.append(test_Y_scaled[i + n_future - 1])

        self.testX, self.testY = np.array(testX), np.array(testY)

        return self.trainX, self.trainY, self.testX, self.testY

    def inverse_target_scaler(self, predictY):
        scalerY = self.scalerY
        predictY_list = scalerY.inverse_transform(predictY)
        predictY_values = []
        for x in predictY_list:
            predictY_values.append(x[0])

        return predictY_values

    def actual_predicted_target_values(self, predictY):
        scalerY = self.scalerY
        predictY_list = scalerY.inverse_transform(predictY)
        predictY_values = []
        for x in predictY_list:
            predictY_values.append(x[0])
        df = self.dfTestSplit[-len(predictY):].copy()
        df['predicted value'] = predictY_values

        return df
