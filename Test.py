import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
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
    df2 = pd.DataFrame([[1, 2, 1], [4, 3, 6]], columns=['col1', 'col2', 'col3'], index=['a', 'b'])
    print(df2)
    print(len(df2[df2['col1']==df2['col3']]))
    # le = LabelEncoder().fit(["A","B","C"])
    # print(le.transform(["A","B","C"]))
    # print(le.transform(["C", "D", "A"]))
    # x, indexer = pd.factorize(["A","B","C"])
    # print(x)
    # y = indexer.get_indexer(['A',"C","B"])
    # print(y)
    # print(OneHotEncoder(sparse=False).fit_transform(["A", "B"]))
    # indexer.get_indexer(["A","B","B"])
    # preProcessing = PreProcessing()
    # testDataFilePath = "../temp/data/test.csv"
    # data = preProcessing.load_data(testDataFilePath)
    # # print(data['device'])
    # # print(data['device']['browser'])
    # print(data['device'].dtype)
    # jsonData = json_normalize(data['device'])
    # # print(jsonData.dtypes)
    # print(jsonData['browser'])
    # print(data['fullVisitorId'])
    # print(data['visitStartTime'])
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