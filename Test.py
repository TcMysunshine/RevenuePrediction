# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
data = {'score':[90, 80, 85]}
df = pd.DataFrame(data)
print(df)
from sklearn.preprocessing import MinMaxScaler
print(MinMaxScaler().fit_transform(df[['score']]))
#
# data1 = {'score':[90, 81, 83]}
# df1 = pd.DataFrame(data1)
# print(df1)
# print(df-df1)
# result =(df-df1)*(df-df1)
#
# print(result)
# print(result.sum()/len(result))
# print("相同")
# print(len(df[df['score'] == df1['score']]))
# train = pd.DataFrame({'sex':['male', 'female', 'other','male']})
# labelEncoder = LabelEncoder().fit(train['sex'])
# test = pd.DataFrame({'sex':['female', 'male', 'male']})
# print(LabelEncoder().fit_transform(train['sex']))
# print(labelEncoder.transform(test['sex']))
# a,indexer = pd.factorize(train['sex'])
# print(a)
# print(indexer.get_indexer(test['sex']))
# print('OneHotEncoder')
# ohe = OneHotEncoder().fit_transform(train[['sex']]).toarray()
# print(ohe)
# print(pd.DataFrame(ohe))
# print('end')
