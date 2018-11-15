import numpy as np
from sklearn.neural_network import MLPRegressor  # 多层线性回归
import pandas as pd

'''神经网络回归'''


class NnMLP:
    def __init__(self):
        self.mlp = MLPRegressor(solver='lbfgs', alpha=1e-5,
                                hidden_layer_sizes=(5, 2), random_state=1)

    def trainModel(self, data, target):
        self.mlp.fit(data, target)

    def predict(self, predictdata):
        return self.mlp.predict(predictdata)


if __name__ == '__main__':
    basePath = "../temp/data/"

    """获取训练和测试集"""
    trainDataFilePath = basePath + "train_cato.csv"
    testDataFilePath = basePath + "test_cato.csv"
    traindata = pd.read_csv(trainDataFilePath, low_memory=False)
    testdata = pd.read_csv(testDataFilePath, low_memory=False)

    '''去除掉fullVisitorId'''
    trainfullVisitorId = traindata.pop("fullVisitorId")
    testfullVisitorId = testdata.pop("fullVisitorId")
    '''过滤出预测集'''
    target = traindata.pop("totals.transactionRevenue")
    '''开始训练预测'''
    mlp = NnMLP()
    mlp.trainModel(traindata, target)
    result = mlp.predict(testdata)
    '''写入文件'''
    resultPath = basePath + "resultNN.csv"
    '''合并转为Dataframe'''
    testfullVisitorId = np.array(testfullVisitorId)
    result = pd.DataFrame(result)
    testfullVisitorIdDf = pd.DataFrame(testfullVisitorId)
    finalResult = pd.concat([testfullVisitorIdDf, result], axis=1)
    '''加上列名'''
    finalResult.columns = ['fullVisitorId', 'PredictedLogRevenue']
    print(finalResult.shape)
    finalResult['fullVisitorId'].astype(str)
    '''groupby并存储'''
    saveResult = finalResult.groupby(finalResult["fullVisitorId"]).sum().reset_index()
    saveResult.to_csv(resultPath, sep=',', header=True, index=False)
    print(saveResult.shape)