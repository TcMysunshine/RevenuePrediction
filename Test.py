import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize


class PreProcessing:
    def __init__(self):
        super().__init__()

    def load_data(self, filePath):
        data = []
        # with open(filePath, 'r') as f:
        #     for rowData in f.readlines():
        #         data.append(rowData)
        # print(data[0])
        # print(data[1])
        # channelGrouping,date,device,fullVisitorId,geoNetwork,totals,sessionId,socialEngagementType,
        # totals,trafficSource,visitId,visitNumber,visitStartTime
        json_columns = ['device', 'geoNetwork', 'totals', 'trafficSource']
        data = pd.read_csv(filePath, dtype={'fullVisitorId': 'str', 'sessionId': 'str',
                                            'socialEngagementType': 'str', 'channelGrouping': 'str'},
                           converters={column: json.loads for column in json_columns})
        return data


if __name__ == '__main__':
    preProcessing = PreProcessing()
    testDataFilePath = "../temp/data/test.csv"
    data = preProcessing.load_data(testDataFilePath)
    # print(data['device'])
    # print(data['device']['browser'])
    print(data['device'].dtype)
    jsonData = json_normalize(data['device'])
    # print(jsonData.dtypes)
    print(jsonData['browser'])
    print(data['fullVisitorId'])
    print(data['visitStartTime'])
    # print(json_normalize(data['device']))
    # device
    # print(data['device'])
    # browse = data['device']
    # print(browse[0])
    # jsonBrowse = json.loads(browse[0])
    # print(jsonBrowse)
    # print(jsonBrowse.get('browser')+","+jsonBrowse['operatingSystem']+","+str(jsonBrowse['isMobile'])+","+jsonBrowse['deviceCategory'])

    # geoNetwork = data['geoNetwork']
    # print(geoNetwork)
    # print(jsonBrowse['operatingSystem'])
    # print(jsonBrowse['isMobile'])
    # print(≈)
    # print(data.info())
    # print(data[[:3],[0:3]])

#通过json.load和json_normalize将json转化为dataFrame