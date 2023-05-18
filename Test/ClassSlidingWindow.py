from sklearn.preprocessing import StandardScaler
import numpy as np

class SlidingWindow(object):
    '''
    '''

    def __init__(self, dataframe, features, target):
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.scaler = StandardScaler()
        self.trainX = 0
        self.trainY = 0
        self.scalerF = 0
        self.scalerT = 0

    def split(self):
        df_features = self.dataframe[self.features]
        scaler = self.scaler
        scalerF = scaler.fit(df_features)
        df_features_scaled = scalerF.transform(df_features)
        self.scalerF = scalerF

        df_target = self.dataframe[self.target]
        scalerT = scaler.fit(df_target)
        df_target_scaled = scalerT.transform(df_target)
        self.scalerT = scalerT

        trainX = []
        trainY = []

        n_future = 1  # Number of days we want to look into the future based on the past days.
        n_past = 3  # Number of past days we want to use to predict the future.

        for i in range(n_past, len(df_features_scaled) - n_future + 1):
            trainX.append(df_features_scaled[i - n_past:i])
            trainY.append(df_target_scaled[i + n_future - 1])

        self.trainX, self.trainY = np.array(trainX), np.array(trainY)
        return self.trainX, self.trainY

    def inverse_scaler(self):
        scalerF = self.scalerF
        trainX = self.trainX
        x = []
        for i in trainX:
            inverseX = scalerF.inverse_transform(x)
            x.append(inverseX)

        scalerT = self.scalerT
        trainY = self.trainY
        y = []
        for i in trainY:
            inverseY = scalerT.inverse_transform(y)
            y.append(inverseY)

        return np.array(x), np.array(y)