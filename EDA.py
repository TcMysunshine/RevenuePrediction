from PreProcessing import PreProcessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class EDA:
    def __init__(self):
        super().__init__()

    def boxplot(self, data):
        # data = pd.DataFrame({
        #     "dataSet1": [1, 2, 9, 6],
        #     "dataSet2": [7, 9, 90, 3],
        #     "dataSet3": [78, 90, 190, 6],
        #     "dataSet4": [78, 9, 0, 87],
        # })
        # draw
        '''最大值，最小值，均值，上四分位，下四分位'''
        data.boxplot()
        plt.ylabel("value")
        plt.xlabel("label")
        plt.show()

    def bar(self, x, y):
        plt.bar(x, y, width=0.35, facecolor='lightskyblue', edgecolor='white')
        plt.show()

    def hist(self, data):
        # data = pd.DataFrame({
        #     "dataSet1": [1, 2, 9, 6],
        #     "dataSet2": [7, 9, 90, 3],
        #     "dataSet3": [78, 90, 190, 6],
        #     "dataSet4": [78, 9, 0, 87],
        # })
        # draw
        """每一列数据的分布图"""
        """normed表示是否将频数转换为频率"""
        data.hist(density=False)
        print(data)
        # plt.tick_params(top='off', right='off')
        # plt.legend()
        plt.ylabel("value")
        plt.xlabel("label")
        plt.show()

    def get_csv_data(self, dataFilePath):
        data = pd.read_csv(dataFilePath)
        print(data.shape)
        return data

    def write_column_count(self, data, column_count_file):
        with open(column_count_file, 'a') as a:
            for column in data.columns.values:
                count = data[column].nunique(dropna=False)
                a.write(column + ":" + str(count) + '\n')

    def get_column_value(self, filePath):
        columns = []
        values = []
        with open(filePath, 'r') as r:
            lines = r.readlines()
        for line in lines:
            line = line.strip().split(":")
            temp_column = line[0]
            temp_value = int(line[1])
            # print(tempColumn + str(tempCount))
            columns.append(temp_column)
            values.append(temp_value)
        return columns, values

    def write_null_column(self, data, column_null_file):
        with open(column_null_file, 'a') as a:
            for column in data.columns.values:
                null_count = data[column].isnull().sum()
                # print(column + ":" + str()))
                a.write(column + ":" + str(null_count) + '\n')


if __name__ == '__main__':
    preProcessing = PreProcessing()
    eda = EDA()
    basePath = "../temp/data/"

    """获取训练和测试集"""
    trainDataFilePath = basePath + "traindata.csv"
    testDataFilePath = basePath + "testdata.csv"
    traindata = eda.get_csv_data(trainDataFilePath)
    testdata = eda.get_csv_data(testDataFilePath)
    datetime = pd.to_datetime(traindata['visitStartTime'], unit='s')
    # print(datetime.days)
    print(traindata.loc[0:5, ['visitStartTime']])
    print(datetime.loc[0:5])
    print(datetime.loc[0:5].dt.year)
    print(datetime.loc[0:5].dt.month)
    print(datetime.loc[0:5].dt.day)
    print(datetime.loc[0:5].dt.weekday)
    print(datetime.loc[0:5].dt.hour)
    # traindata['totals.newVisits']
    # exclued_features = ['trafficSource.adwordsClickInfo.adNetworkType', 'trafficSource.adwordsClickInfo.isVideoAd',
    #                     'trafficSource.adwordsClickInfo.page', 'trafficSource.adwordsClickInfo.slot']
    # for feature in exclued_features:
    #     print(feature)
    #     print(traindata[feature].unique())

    # print(traindata['trafficSource.campaignCode'].unique())

    # """将每一列属性每个值出现的次数写入文件"""
    # eda.write_column_count(traindata, basePath + "train_column_count.txt")
    # eda.write_column_count(testdata, basePath + "test_column_count.txt")
    #
    # """将每一列属性空值的个数写入文件"""
    # eda.write_null_column(traindata, basePath + "train_column_null.txt")
    # eda.write_null_column(testdata, basePath + "test_column_null.txt")

    # """填补空值"""
    # null_features = ['totals.bounces', 'totals.newVisits', 'totals.transactionRevenue',
    #                  'totals.pageviews', 'trafficSource.adContent',
    #                  'trafficSource.adwordsClickInfo.adNetworkType',
    #                  'trafficSource.adwordsClickInfo.gclId', 'trafficSource.adwordsClickInfo.isVideoAd',
    #                  'trafficSource.adwordsClickInfo.page', 'trafficSource.adwordsClickInfo.slot',
    #                  'trafficSource.isTrueDirect', 'trafficSource.keyword',
    #                  'trafficSource.referralPath']
    #
    # str_null_features = ['trafficSource.adContent', 'trafficSource.adwordsClickInfo.adNetworkType',
    #                      'trafficSource.adwordsClickInfo.gclId', 'trafficSource.adwordsClickInfo.slot',
    #                      'trafficSource.keyword', 'trafficSource.referralPath']
    #
    # numeric_null_features =['trafficSource.adwordsClickInfo.page', 'totals.pageviews',
    #                         'totals.newVisits', 'totals.bounces']
    #
    # # boolean_null_features = ['trafficSource.isTrueDirect', 'trafficSource.adwordsClickInfo.isVideoAd']
    # target_features = ['totals.transactionRevenue']
    #
    # for str_feature in str_null_features:
    #     traindata[str_feature].fillna('Null', inplace=True)
    # for num_feature in numeric_null_features:
    #     traindata[num_feature].fillna(0, inplace=True)
    # traindata['device.isMobile'].replace({True: 1, False: 0}, inplace=True)
    # traindata['trafficSource.adwordsClickInfo.isVideoAd'].replace({np.nan: 1, False: 0}, inplace=True)
    # traindata['trafficSource.isTrueDirect'].replace({np.nan: 0, True: 1}, inplace=True)
    # traindata.to_csv('../temp/data/traindata_v1.csv', sep=',', header=True, index=False)




    # print("测试集没有的属性")
    # for trainColumn in trainColumns:
    #     if trainColumn not in testColumns:
    #         print(trainColumn)
    # print("训练集没有的属性")
    # for testColumn in testColumns:
    #     if testColumn not in trainColumns:
    #         print(testColumn)


    # eda.bar(trainColumns, trainDistribution)
    # eda.hist(traindata['channelGrouping'])
    # print(trainData[['fullVisitorId']])
    # eda.hist(traindata.loc[:, 'device.browser'])



