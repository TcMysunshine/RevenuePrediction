from PreProcessing import PreProcessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

class EDA:

    '''箱形图，通常用来确认数据的分布情况，如最大值最小值，可用来确定异常值'''
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

    '''柱状图'''
    def bar(self, x, y):
        plt.bar(x, y, width=0.35, facecolor='lightskyblue', edgecolor='white')
        plt.show()

    '''确定每一列数据的分布情况'''
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

    ''' 读取CSV文件'''
    def get_csv_data(self, dataFilePath):
        data = pd.read_csv(dataFilePath, low_memory=False)
        print(data.shape)
        return data

    '''将每一列所拥有的类别个数写入文件'''
    def write_column_count(self, data, column_count_file):
        with open(column_count_file, 'a') as a:
            for column in data.columns.values:
                count = data[column].nunique(dropna=False)
                a.write(column + ":" + str(count) + '\n')

    '''获取键值对'''
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

    '''将每一列空值个数写入文件'''
    def write_null_column(self, data, column_null_file):
        with open(column_null_file, 'a') as a:
            for column in data.columns.values:
                null_count = data[column].isnull().sum()
                # print(column + ":" + str()))
                a.write(column + ":" + str(null_count) + '\n')

    '''将某一列的某一个值和所对应的个数画出图形'''
    def draw_value_count(self, data, column):
        value_count = data[column].value_counts()
        print(value_count)
        value_count.plot(kind="bar", title="value distri",
                         figsize=(8, 8), rot=25, colormap='Paired')
        plt.ylabel("count")
        plt.xlabel("value")
        plt.show()


if __name__ == '__main__':
    preProcessing = PreProcessing()
    eda = EDA()
    basePath = "../temp/data/"

    """获取训练和测试集"""
    trainDataFilePath = basePath + "train_fillna.csv"
    testDataFilePath = basePath + "test_fillna.csv"
    traindata = eda.get_csv_data(trainDataFilePath)
    testdata = eda.get_csv_data(testDataFilePath)
    '''查看visitID与visitStartTime相似程度'''
    # print(len(traindata[traindata['visitStartTime']==traindata['visitId']]))
    # print(len(testdata[testdata['visitStartTime'] == testdata['visitId']]))
    '''某一列的value-count Start'''
    # print(traindata['trafficSource.campaignCode'].value_counts())
    print(traindata['device.deviceCategory'].value_counts())

    eda.draw_value_count(traindata, 'channelGrouping')
    '''value-count END'''
    # numer_columns = ['visitNumber', 'totals.bounces',
    #                  'totals.hits', 'totals.newVisits',
    #                  'totals.pageviews', 'trafficSource.adwordsClickInfo.page']
    # data = traindata.loc[:, numer_columns]
    # # print(data)
    # eda.hist(data)

    # for c in numer_columns:
    #     print(data[c].value_counts())
    '''时间转换'''
    # visitStartTime = pd.to_datetime(traindata['visitStartTime'], unit="s")
    # date = pd.to_datetime(traindata['date'].astype(str), format="%y%m%d", errors='ignore')
    # print(date.loc[0:5].dt.year)
    # print(date.loc[0:5].dt.month)
    # print(date.loc[0:5].dt.day)
    # print(traindata.loc[0:5, "visitStartTime"])
    # print(visitStartTime.loc[0:5].astype(str))
    # # print(visitStartTime.loc[0:5].astype(str))
    # temp = date.loc[0:5]
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




