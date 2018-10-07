from PreProcessing import PreProcessing
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd


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
        data.to_csv(saveFilePath, sep=',', header=True, index=False)

    """计算log(1+x) 底数为e"""
    def log1p(self, data):
        return np.log1p(data)

    '''将类别数据转换为oneHotEncode'''
    def get_dummpy(self, data, columns):
        return pd.get_dummies(data, columns=columns)

    """训练集中将类别数据进行划分之后，运用到测试集中"""
    def catogory_train_test(self, train, test, categorical_features):
        for f in categorical_features:
            train[f], indexer = pd.factorize(train[f])
            test[f] = indexer.get_indexer(test[f])
        return train, test


if __name__ == '__main__':
    featureEngineer = FeatureEngineer()
    basePath = "../temp/data/"
