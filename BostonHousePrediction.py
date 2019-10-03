import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold

standardScaler = StandardScaler()
linearRegression = LinearRegression()
decisionTreeRegressor = DecisionTreeRegressor()
randomForestRegressor = RandomForestRegressor()
supportVectorRegressor = SVR(kernel='rbf', gamma=0.01)

data = pd.read_csv('data/boston_housing.data', header=None, delimiter=' ')

modelDataframe = pd.DataFrame({
        'name': ['LinearRegression', 'DecisionTreeRegressor',
        'RandomForestRegressor', 'SupportVectorRegressor'], 
        'model': [linearRegression, decisionTreeRegressor,
        randomForestRegressor, supportVectorRegressor]
    })

x = data[range(13)]
y = data[13] * 1000

#x_train, x_test, y_train, y_test = train_test_split(x, y, 
    #test_size=0.3, random_state=7)

#x_train = standardScaler.fit_transform(x_train)
#x_test = standardScaler.fit_transform(x_test)

kFold = KFold(n_splits=10, random_state=7)

for index, row in modelDataframe.iterrows():
    xval_err = 0
    for train, test in kFold.split(x):
        row['model'].fit(x.iloc[train], y[train])
        prediction = row['model'].predict(x.iloc[test])
        print(prediction)
        err = prediction - y[test]
        xval_err += np.dot(err, err)
    rmse_10cv = np.sqrt(xval_err/len(x))
    print(rmse_10cv)
    #row['model'].fit(x_train, y_train)
    #prediction = row['model'].predict(x_test)
    #print('Name:', row['name'])
    #print(prediction)
