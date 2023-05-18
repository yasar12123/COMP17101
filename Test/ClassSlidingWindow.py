from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class SlidingWindow(object):
    '''
    '''

    def __init__(self, dataframe, features, target):
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.scaler = StandardScaler()
        self.trainX = []
        self.trainY = []
        self.testX = []
        self.testY = []
        self.scaler_trainX = 0
        self.scaler_trainY = 0
        self.scaler_testX = 0
        self.scaler_testY = 0

    def split(self, split_ratio, steps_back, steps_forward):
        #train split
        split = int(len(self.dataframe) * split_ratio)
        train_split = self.dataframe[:split]
        train_X = train_split[self.features]
        train_Y = train_split[self.target]

        #scale data
        scaler = self.scaler
        # scale trainX
        scaler_trainX = scaler.fit(train_X)
        train_X_scaled = scaler_trainX.transform(train_X)
        self.scaler_trainX = scaler_trainX
        # scale trainY
        scaler_trainY = scaler.fit(train_Y)
        train_Y_scaled = scaler_trainY.transform(train_Y)
        self.scaler_trainY = scaler_trainY

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
        test_X = test_split[self.features]
        test_Y = test_split[self.target]

        #scale data
        scaler = self.scaler
        # scale testX
        scaler_testX = scaler.fit(test_X)
        test_X_scaled = scaler_testX.transform(test_X)
        self.scaler_testX = scaler_testX
        # scale trainY
        scaler_testY = scaler.fit(test_Y)
        test_Y_scaled = scaler_testY.transform(test_Y)
        self.scaler_testY = scaler_testY

        #split test data into steps
        testX = []
        testY = []
        for i in range(n_past, len(test_X_scaled) - n_future + 1):
            testX.append(test_X_scaled[i - n_past:i])
            testY.append(test_Y_scaled[i + n_future - 1])

        self.testX, self.testY = np.array(testX), np.array(testY)

        return self.trainX, self.trainY, self.testX, self.testY


    # def inverse_scaler(self):
    #     scalerF = self.scalerF
    #     trainX = self.trainX
    #     x = []
    #     for i in trainX:
    #         inverseX = scalerF.inverse_transform(trainX)
    #         x.append(inverseX)
    #
    #     # scalerT = self.scalerT
    #     # trainY = self.trainY
    #     # y = []
    #     # for i in trainY:
    #     #     inverseY = scalerT.inverse_transform(y)
    #     #     y.append(inverseY)
    #
    #     return np.array(x)#, np.array(y)