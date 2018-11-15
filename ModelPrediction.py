from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
'''随机森林'''


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
        self.rfr = RandomForestRegressor(n_estimators=10, max_depth=None,
                                         max_features='auto', min_samples_split=1,
                                         min_samples_leaf=1, oob_score=False,
                                         random_state=None, n_jobs=1)

    def trainModel(self, data, target):
        self.rfr.fit(data, target)

    def predict(self, predictdata):
        return self.rfr.predict(predictdata)


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
    rfr = RandomForestRegress()
    rfr.trainModel(traindata, target)
    result = rfr.predict(testdata)
    '''写入文件'''
    resultPath = basePath + "resultRFR.csv"
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