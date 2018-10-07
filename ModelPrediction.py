from sklearn.ensemble import RandomForestRegressor
# from sklearn.emsemble import RandomForestClassifier
import numpy as np
import pandas as pd
X = pd.DataFrame({'age': [1, 2, 3], 'salary': [0, 1, 0]})
Y = pd.DataFrame({'height': [170, 176, 156]})
# rfc = RandomForestClassifier()
rfr = RandomForestRegressor(n_estimators=10)
rfr.fit(X, Y['height'])
print(rfr.predict([[32, 45]]))
# rfr.predict()
