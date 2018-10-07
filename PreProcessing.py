import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize
from sklearn.preprocessing import LabelEncoder
'''预处理'''


class PreProcessing:
    def __init__(self):
        super().__init__()

    '''读取数据集'''
    def load_data(self, filePath):
        """ channelGrouping,date,device,fullVisitorId,geoNetwork,totals,sessionId,
        socialEngagementType,totals,trafficSource,visitId,visitNumber
        """
        '''数据为Json的数据列'''
        json_columns = ['device', 'geoNetwork', 'totals', 'trafficSource']
        data = pd.read_csv(filePath, low_memory=False,
                           dtype={'fullVisitorId': 'str', 'sessionId': 'str',
                                  'socialEngagementType': 'str', 'channelGrouping': 'str'},
                           converters={column: json.loads for column in json_columns})
        for column in json_columns:
            # print(column)
            '''运用json.load和json_normalize将json转换为DataFrame'''
            json_normalized_data = json_normalize(data[column])
            '''列名加键名作为列名'''
            json_normalized_data.columns = [f'{column}.{subColumn}' for subColumn in json_normalized_data.columns]
            '''索引作为连接键'''
            data = data.drop(column, axis=1).merge(json_normalized_data, right_index=True, left_index=True)
        return data
    # '''获取每一列每个值出现的次数'''
    # def get_unique_column_value(self, data):
    #     unique_values_count = {}
    #     # unique_values = {}
    #     for c in data.columns.values:
    #         unique_values_count[c] = data[c].nunique(dropna=False)
    #         # unique_values[c] = data[c].unique
    #     return unique_values_count
        # unique_values=[c for c in data.columns if data[c].nunique(dropna=False)]

    '''消除某一列数据为常量的值'''
    def remove_constant_columns(self, data):
        constant_columns = [c for c in data.columns.values if data[c].nunique(dropna=False) == 1]
        result = data.drop(constant_columns, axis=1, inplace=False)
        return result

    '''将属性为类别的转换为编码'''
    def class2label(self, classValue):
        dle = LabelEncoder()
        '''获取这一列数据的有哪些类别'''
        print(classValue)
        total_class_value = np.unique(classValue)
        print(total_class_value)
        labels = dle.fit_transform(classValue)
        mappings = {index: label for index, label in enumerate(dle.classes_)}
        print(labels)
        print(mappings)

    def write2csv(self, filePath, toFilePath):
        # DataFilePath = "../temp/data/train.csv"
        data = self.load_data(filePath)
        # print()
        # print(data.columns.values)
        # print(testdata.columns)
        print(data.shape)
        '''消除某一列数据为常量的值'''
        data = preProcessing.remove_constant_columns(data)
        print(data.columns.values)
        print(data.shape)
        """处理后的数据写入csv文件"""
        '''header 表示是否需要表头，index表示是否需要行号'''
        data.to_csv(toFilePath, sep=',', header=True, index=False)


if __name__ == '__main__':
    preProcessing = PreProcessing()
    DataFilePath = "../temp/data/test.csv"
    data = preProcessing.load_data(DataFilePath)
    # print()
    print(data.columns.values)
    # print(testdata.columns)
    print(data.shape)
    # unique_columnvalue = preProcessing.get_unique_column_value(data)
    # for key, value in unique_columnvalue.items():
    #     print(key+":"+str(value))
    '''消除某一列数据为常量的值'''
    data = preProcessing.remove_constant_columns(data)
    # print(data.columns.values)
    print(data.shape)
    """处理后的数据写入csv文件"""
    '''header 表示是否需要表头，index表示是否需要行号'''
    data.to_csv('../temp/data/testdata.csv', sep=',', header=True, index=False)
    # readdata = pd.read_csv('../temp/data/traindata.csv')
    # print(readdata)
    # preProcessing.class2label(readdata['device.browser'])
    # print('Category')
    # print(readdata['device.browser'].astype('category'))
    # trainDataFilePath = "../temp/data/train.csv"
    # traindata = preProcessing.load_data(trainDataFilePath)
    # print()
    # print(traindata.columns.values)
    # print(traindata.shape)
