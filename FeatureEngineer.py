from PreProcessing import PreProcessing
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import imp

class FeatureEngineer:
    def fillNa(self, data, saveFilePath):
        """填补空值"""
        null_features = ['totals.bounces', 'totals.newVisits', 'totals.transactionRevenue',
                         'totals.pageviews', 'trafficSource.adContent',
                         'trafficSource.adwordsClickInfo.adNetworkType',
                         'trafficSource.adwordsClickInfo.gclId', 'trafficSource.adwordsClickInfo.isVideoAd',
                         'trafficSource.adwordsClickInfo.page', 'trafficSource.adwordsClickInfo.slot',
                         'trafficSource.isTrueDirect', 'trafficSource.keyword',
                         'trafficSource.referralPath']

        str_null_features = ['trafficSource.adContent', 'trafficSource.adwordsClickInfo.adNetworkType',
                             'trafficSource.adwordsClickInfo.gclId', 'trafficSource.adwordsClickInfo.slot',
                             'trafficSource.keyword', 'trafficSource.referralPath']

        numeric_null_features =['trafficSource.adwordsClickInfo.page', 'totals.pageviews',
                                'totals.newVisits', 'totals.bounces']

        # boolean_null_features = ['trafficSource.isTrueDirect', 'trafficSource.adwordsClickInfo.isVideoAd']

        '''删除掉无用值'''
        campaignCode = "trafficSource.campaignCode"
        if  campaignCode in data.columns:
            del data[campaignCode]

        """将需要预测的转换为对数"""
        target_features = 'totals.transactionRevenue'
        if target_features in data.columns:
            data[target_features].fillna(0, inplace=True)
            data[target_features] = np.log1p(data[target_features])

        '''填补类别为字符串的'''
        for str_feature in str_null_features:
            data[str_feature].fillna('Null', inplace=True)

        '''填补类型为数值'''
        for num_feature in numeric_null_features:
            data[num_feature].fillna(0, inplace=True)

        '''填补类型为boolean'''
        data['device.isMobile'].replace({True: 1, False: 0}, inplace=True)
        data['trafficSource.adwordsClickInfo.isVideoAd'].replace({np.nan: 1, False: 0}, inplace=True)
        data['trafficSource.isTrueDirect'].replace({np.nan: 0, True: 1}, inplace=True)

        '''保存到csv中'''
        data.to_csv(saveFilePath, sep=',', header=True, index=False)

    def feature_transform(self, data, saveFilePath):
        '''日期转换'''
        datetime = pd.to_datetime(data['visitStartTime'], unit='s')
        data['year'] = datetime.dt.year
        data['month'] = datetime.dt.month
        data['day'] = datetime.dt.day
        data['weekday'] = datetime.dt.weekday
        data['hour'] = datetime.dt.hour
        del data['visitStartTime']

        '''由于visitStartTime与visitId值相同的概率达到了百分之九十九因此删除掉visitID一列'''
        del data['visitId']
        '''删除掉sessionID'''
        del data['sessionId']
        '''保存到csv中'''
        data.to_csv(saveFilePath, sep=',', header=True, index=False)
    # """计算log(1+x) 底数为e"""
    # def log1p(self, data):
    #     return np.log1p(data)

    # '''将类别数据转换为oneHotEncode'''
    # def get_dummpy(self, data, columns):
    #     return pd.get_dummies(data, columns=columns)

    """训练集中将类别数据进行划分之后，运用到测试集中"""
    def catogory_train_test_encode(self, train, test, trainFilePath,testFilePath, categorical_features):
        for f in categorical_features:
            '''将类别数据转换为数据编码'''
            train[f], indexer = pd.factorize(train[f])
            # '''将该列转换为onehotencode'''
            # oheTrain = OneHotEncoder(sparse=False).fit_transform(train[[f]])
            # '''删除该列'''
            # train.pop(f)
            # '''填补上该列数据'''
            # train = np.hstack((train, oheTrain))
            # print(train)
            # """测试数据"""
            test[f] = indexer.get_indexer(test[f])
            # print(test[f])
            # oheTest = OneHotEncoder(sparse=False).fit_transform(test[[f]])
            # test.pop(f)
            # test = np.hstack((test, oheTest))
        train.to_csv(trainFilePath, sep=',', header=True, index=False)
        test.to_csv(testFilePath, sep=',', header=True, index=False)


    # """训练集中将类别数据进行划分之后，运用到测试集中"""
    #
    # def catogory_train_test(self, train, test, categorical_features):
    #     for f in categorical_features:
    #         '''将类别数据转换为数据编码'''
    #         x, indexer_train = pd.factorize(train[f])
    #         # test[f], indexer_test = pd.factorize(test[f])
    #         y = indexer_train.get_indexer(test[f])
    #         print(x)
    #         print("hh")
    #         print(y)
    #         print(OneHotEncoder(sparse=False, categories='auto').fit_transform(x.reshape(-1,1)))
    #         print(OneHotEncoder(sparse=False, categories='auto').fit_transform(y.reshape(-1,1)))
    #     return train, test


if __name__ == '__main__':
    featureEngineer = FeatureEngineer()
    basePath = "../temp/data/"
    '''去除空值Start'''
    # tranDataPath = basePath + "traindata.csv"
    # testDataPath = basePath + "testdata.csv"

    # data = pd.read_csv(tranDataPath, low_memory=False)
    # featureEngineer.fillNa(data, basePath + "train_fillna.csv")
    #
    # testdata = pd.read_csv(testDataPath, low_memory=False)
    # featureEngineer.fillNa(testdata, basePath + "test_fillna.csv")
    '''去除空值End'''

    # '''去掉不必要的数据和日期转换Start'''
    # tranDataPath = basePath + "train_fillna.csv"
    # testDataPath = basePath + "test_fillna.csv"
    #
    # data = pd.read_csv(tranDataPath, low_memory=False)
    # featureEngineer.feature_transform(data, basePath + "train_featureTran.csv")
    #
    # testdata = pd.read_csv(testDataPath, low_memory=False)
    # featureEngineer.feature_transform(testdata, basePath + "test_featureTran.csv")
    # '''去掉不必要的数据和日期转换End'''

    '''类别数据转换为数值型Start'''
    tranDataPath = basePath + "train_featureTran.csv"
    testDataPath = basePath + "test_featureTran.csv"

    traindata = pd.read_csv(tranDataPath, low_memory=False)
    testdata = pd.read_csv(testDataPath, low_memory=False)
    trainFilepath = basePath + "train_cato.csv"
    testFilepath = basePath + "test_cato.csv"
    catoFeatures = ["channelGrouping","device.browser","device.deviceCategory",
                    "device.operatingSystem","geoNetwork.city","geoNetwork.continent",
                    "geoNetwork.country", "geoNetwork.metro", "geoNetwork.networkDomain",
                    "geoNetwork.region", "geoNetwork.subContinent","trafficSource.adContent",
                    "trafficSource.adwordsClickInfo.adNetworkType", "trafficSource.adwordsClickInfo.gclId",
                    "trafficSource.adwordsClickInfo.slot","trafficSource.campaign","trafficSource.keyword",
                    "trafficSource.medium","trafficSource.referralPath","trafficSource.source"]
    featureEngineer.catogory_train_test_encode(traindata, testdata, trainFilepath, testFilepath, catoFeatures)
    '''类别数据转换为数值型End'''

    # traindata_all_features = ['channelGrouping','date','fullVisitorId',
    #                           'sessionId', 'visitId', 'visitNumber',
    #                           'visitStartTime', 'device.browser', 'device.deviceCategory',
    #                           'device.isMobile', 'device.operatingSystem',
    #                           'geoNetwork.city', 'geoNetwork.continent',
    #                           'geoNetwork.country', 'geoNetwork.metro', 'geoNetwork.networkDomain',
    #                           'geoNetwork.region', 'geoNetwork.subContinent', 'totals.bounces',
    #                           'totals.hits', 'totals.newVisits', 'totals.pageviews',
    #                           'totals.transactionRevenue','trafficSource.adContent',
    #                           'trafficSource.adwordsClickInfo.adNetworkType'
    #                           'trafficSource.adwordsClickInfo.gclId',
    #                           'trafficSource.adwordsClickInfo.isVideoAd',
    #                           'trafficSource.adwordsClickInfo.page',
    #                           'trafficSource.adwordsClickInfo.slot',
    #                           'trafficSource.campaign',
    #                           'trafficSource.isTrueDirect',
    #                           'trafficSource.keyword',
    #                           'trafficSource.medium',
    #                           'trafficSource.referralPath',
    #                           'trafficSource.source'
    #                          ]
    # testdata_all_features = ['channelGrouping', 'date', 'fullVisitorId',
    #                           'sessionId', 'visitId', 'visitNumber',
    #                           'visitStartTime', 'device.browser', 'device.deviceCategory',
    #                           'device.isMobile', 'device.operatingSystem',
    #                           'geoNetwork.city', 'geoNetwork.continent',
    #                           'geoNetwork.country', 'geoNetwork.metro', 'geoNetwork.networkDomain',
    #                           'geoNetwork.region', 'geoNetwork.subContinent',
    #                           'totals.bounces', 'totals.hits', 'totals.newVisits', 'totals.pageviews',
    #                           'trafficSource.adContent',
    #                           'trafficSource.adwordsClickInfo.adNetworkType'
    #                           'trafficSource.adwordsClickInfo.gclId',
    #                           'trafficSource.adwordsClickInfo.isVideoAd',
    #                           'trafficSource.adwordsClickInfo.page',
    #                           'trafficSource.adwordsClickInfo.slot',
    #                           'trafficSource.campaign',
    #                           'trafficSource.isTrueDirect',
    #                           'trafficSource.keyword',
    #                           'trafficSource.medium',
    #                           'trafficSource.referralPath',
    #                           'trafficSource.source'
    #                           ]
    # train = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['a', 'b', 'c'], 'C': [1, 2, 3]})
    # # print(pd.get_dummies(train))
    # test = pd.DataFrame({'A': ['d', 'b', 'c'], 'B': ['a', 'b', 'd'], 'C': [1, 2, 3]})
    #
    # train, test = featureEngineer.catogory_train_test(train, test, ['A', 'B'])
    # print(train)
    # print(test)