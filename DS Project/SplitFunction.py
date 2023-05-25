from sklearn.preprocessing import MinMaxScaler
import numpy as np

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
        scalerF = MinMaxScaler(feature_range=(0, 1))
        scalerX = scalerF.fit(train_X)
        scalerT = MinMaxScaler(feature_range=(0, 1))
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




