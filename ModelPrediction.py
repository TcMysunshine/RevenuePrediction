from sklearn.ensemble import RandomForestRegressor
# from sklearn.emsemble import RandomForestClassifier
import numpy as np
import pandas as pd

# Y = pd.DataFrame({'height': [170, 176, 156]})
# rfc = RandomForestClassifier()


class RandomForestRegress:

    '''随机森林回归树调优参数'''
    """max_features: Auto/None,sqrt,0.3 随机森林允许单个决策树使用特征的最大数量
       max_depth
       n_estimators: 子树的个数
       min_samples_leaf： 最小样本叶片大小
       min_samples_split
       n_jobs：处理器的个数 -1表示没有限制 1表示只有一个处理器
       random_state：在参数和训练数据不变的情况下，一个确定的随机值将会产生相同的结果，
       oob_score：交叉验证方法
    """
    def __init__(self):
        self.rfr = RandomForestRegressor()
        # self.data = data
        # self.target = target

    def trainModel(self, data, target):
        self.rfr = RandomForestRegressor(n_estimators=10, max_depth=None, max_features='auto', min_samples_split=2,
                                         min_samples_leaf=1, oob_score=False, random_state=None, n_jobs=1)
        self.rfr.fit(data, target)

    def predict(self, predictdata):
        return self.rfr.predict(predictdata)
# rfr.predict()


if __name__=='__main__':
    # data = pd.DataFrame({'age': [1, 2, 3], 'salary': [0, 1, 0], 'height': [170, 176, 156]})
    # X = data.drop('height', axis=1, inplace=False)
    # Y = data.pop('height')
    # rfr = RandomForestRegress()
    # rfr.trainModel(X, Y)
    # result = rfr.predict([[2, 0], [3, 4]])
    # print(type(result))
    # print(result)
    # print(result.shape)
    # resultDf = pd.DataFrame(result)
    # resultDf.columns = ['transactionRevenue']
    # # resultDf.rename(columns={'0': 'transactionRevenue'}, inplace=True)
    # print(resultDf)
    # b = pd.Series([2, 3])
    # bdf = pd.DataFrame(np.array(b))
    # bdf.columns=['fullVisitorId']
    # print(bdf)
    # print(pd.concat([resultDf,bdf],axis=1))
    basePath = "../temp/data/"

    """获取训练和测试集"""
    trainDataFilePath = basePath + "train_cato.csv"
    testDataFilePath = basePath + "test_cato.csv"
    traindata = pd.read_csv(trainDataFilePath, low_memory=False)
    testdata = pd.read_csv(testDataFilePath, low_memory=False)
    rfr = RandomForestRegress()

    target = traindata.pop("totals.transactionRevenue")
    trainfullVisitorId = traindata.pop("fullVisitorId")
    testfullVisitorId = testdata.pop("fullVisitorId")
    rfr.trainModel(traindata, target)

    result = rfr.predict(testdata)
    resultPath = basePath + "result.csv"
    #
    # print(type(result))
    # print(len(result))
    # print("id")
    # print(type(testfullVisitorId))
    # print(len(testfullVisitorId))
    # # print(result.shape)
    testfullVisitorId = np.array(testfullVisitorId)
    result = pd.DataFrame(result)
    testfullVisitorIdDf = pd.DataFrame(testfullVisitorId)
    finalResult = pd.concat([testfullVisitorIdDf, result], axis=1)
    finalResult.columns = ['fullVisitorId', 'PredictedLogRevenue']
    saveResult = finalResult.groupby(["fullVisitorId"]).sum().reset_index()
    print(saveResult.loc[0:5])
    saveResult.to_csv(resultPath, sep=',', header=True, index=False)