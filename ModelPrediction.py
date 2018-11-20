# -*- coding:utf-8 -*-
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
# from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor


'''计算均方误差'''


def RMSE(y_pred,y_real):
    diff = y_pred-y_real
    mseresult = diff*diff
    return math.sqrt(mseresult.sum()/len(result))


'''转换为OneHotEncoder'''


def ohe(traindata, columns):
    for c in columns:
        channel = pd.DataFrame(OneHotEncoder().fit_transform(traindata[[c]]).toarray())
        value_count = traindata[c].nunique(dropna=False)
        columns_list = []
        for i in range(value_count):
            columns_list.append(c + "-" + str(i))
        channel.columns = columns_list
        print(channel.columns)
        # channel.columns = ['channel-0','channel-1','channel-2']
        traindata = traindata.merge(channel, right_index=True, left_index=True)
        traindata.pop(c)
        print(traindata.shape)
    return traindata


'''数值标准化'''


def min_max_scale(traindata, columns):
    for c in columns:
        traindata[c] = MinMaxScaler().fit_transform(traindata[[c]])
    return traindata


if __name__ == '__main__':
    basePath = "../temp/data/"

    """获取训练和测试集"""
    trainDataFilePath = basePath + "train_cato.csv"
    # testDataFilePath = basePath + "test_cato.csv"
    traindata = pd.read_csv(trainDataFilePath, low_memory=False)
    # testdata = pd.read_csv(testDataFilePath, low_memory=False)
    # print(traindata.shape[1])
    '''去除掉fullVisitorId'''
    trainfullVisitorId = traindata.pop("fullVisitorId")
    # testfullVisitorId = testdata.pop("fullVisitorId")
    '''过滤出预测集'''
    target = traindata.pop("totals.transactionRevenue")

    del traindata['trafficSource.adwordsClickInfo.gclId']
    del traindata['trafficSource.keyword']
    del traindata['trafficSource.referralPath']
    # print(traindata.shape[1])
    yearList = []
    monthList = []
    dayList = []
    for tempdate in traindata['date'].astype(str):
        # print(tempdate[0:4] + tempdate[4:6] + tempdate[6:8])
        yearList.append(int(tempdate[0:4]))
        monthList.append(int(tempdate[4:6]))
        dayList.append(int(tempdate[6:8]))
    x = pd.DataFrame(list(zip(pd.Series(yearList),pd.Series(monthList), pd.Series(dayList))))
    x.columns = ['dateYear', 'dateMonth', 'dateDay']
    # x.

    result = traindata.merge(x, right_index=True, left_index=True)
    print(result.shape)
    del result['dateYear']
    del result['dateMonth']
    del result['date']
    traindata = result
    # traindata['year'] = MinMaxScaler().fit_transform(traindata[['year']])
    # print(traindata['year'])
    print(traindata.shape)

    # '''OneHotEncoder'''
    # traindata = ohe(traindata, ['channelGrouping',
    #                             'device.browser', 'device.deviceCategory', 'device.isMobile', 'device.operatingSystem',
    #                             'geoNetwork.continent', 'geoNetwork.subContinent', 'geoNetwork.metro',
    #                             'trafficSource.adwordsClickInfo.adNetworkType',
    #                             'trafficSource.adwordsClickInfo.slot', 'trafficSource.adwordsClickInfo.isVideoAd',
    #                             'trafficSource.adwordsClickInfo.page', 'trafficSource.adContent',
    #                             'trafficSource.isTrueDirect', 'trafficSource.medium', 'trafficSource.campaign'])
    #
    # '''归一化'''
    # traindata = min_max_scale(traindata, ['geoNetwork.city','geoNetwork.country', 'geoNetwork.networkDomain',
    #                                       'geoNetwork.region', 'trafficSource.source'])
    #
    # '''PCA降维'''
    # print('PCA降维')
    # traindata = PCA(n_components=0.8, svd_solver='full').fit_transform(traindata)


    print(traindata.shape)
    '''开始训练预测'''
    X_train, X_test, Y_train, Y_test = train_test_split(traindata, target, test_size=0.33, random_state=0)

    '''随机森林'''
    rfr = RandomForestRegressor(n_estimators=10, max_depth=None,
                                max_features='auto', min_samples_split=3,
                                min_samples_leaf=1, oob_score=False,
                                random_state=None, n_jobs=1).fit(X_train, Y_train)

    y_pred = rfr.predict(X_test)
    #
    '''神经网络'''
    # mlp = MLPRegressor(solver='lbfgs', alpha=1e-5,
    #                    hidden_layer_sizes=(5, 2), random_state=1).fit(X_train, Y_train)
    # y_pred = mlp.predict(X_test)

    train_size = len(X_train)
    test_size = len(X_test)
    y_pred_df = pd.DataFrame(y_pred)
    y_test_df = pd.DataFrame(Y_test, index=None)
    y_pred_df.rename(columns={0: 'totals.transactionRevenue'}, inplace=True)
    y_test_df.index = range(test_size)
    print(y_pred_df.shape)
    print(y_test_df.shape)
    rmse = RMSE(y_pred_df, y_test_df)
    print(rmse)






    # y_pred = ppn.predict(X_test_std)
    # # 计算模型在测试集上的准确性
    # accuracy_score(y_test, y_pred)
    # result = rfr.predict(testdata)
    # '''写入文件'''
    # resultPath = basePath + "resultRFR.csv"
    # '''合并转为Dataframe'''
    # testfullVisitorId = np.array(testfullVisitorId)
    # result = pd.DataFrame(result)
    # testfullVisitorIdDf = pd.DataFrame(testfullVisitorId)
    # finalResult = pd.concat([testfullVisitorIdDf, result], axis=1)
    # '''加上列名'''
    # finalResult.columns = ['fullVisitorId', 'PredictedLogRevenue']
    # print(finalResult.shape)
    # finalResult['fullVisitorId'].astype(str)
    # '''groupby并存储'''
    # saveResult = finalResult.groupby(finalResult["fullVisitorId"]).sum().reset_index()
    # saveResult.to_csv(resultPath, sep=',', header=True, index=False)
    # print(saveResult.shape)