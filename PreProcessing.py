import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize


class PreProcessing:
    def __init__(self):
        super().__init__()

    def load_data(self, filePath):
        ''' channelGrouping,date,device,fullVisitorId,geoNetwork,totals,sessionId,
        socialEngagementType,totals,trafficSource,visitId,visitNumber'''
        '''数据为Json的数据列'''
        json_columns = ['device', 'geoNetwork', 'totals', 'trafficSource']
        data = pd.read_csv(filePath, dtype={'fullVisitorId': 'str', 'sessionId': 'str',
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


if __name__ == '__main__':
    preProcessing = PreProcessing()
    testDataFilePath = "../temp/data/test.csv"
    testdata = preProcessing.load_data(testDataFilePath)
    # print()
    print(testdata.columns.values)
    print(testdata.shape)

    trainDataFilePath = "../temp/data/train.csv"
    traindata = preProcessing.load_data(trainDataFilePath)
    # print()
    print(traindata.columns.values)
    print(traindata.shape)
