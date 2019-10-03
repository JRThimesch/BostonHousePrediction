import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

boston = pd.read_csv('data/boston_housing.data', header=None, delimiter=' ')
x = boston[range(13)]
y = boston[13]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=7)

linearRegression = LinearRegression()
decisionTreeRegressor = DecisionTreeRegressor()
randomForestRegressor = RandomForestRegressor()
supportVectorRegressor = SVR(kernel='rbf', gamma=0.01)

linearRegression.fit(x_train, y_train)
prediction = linearRegression.predict(x_test)
print(prediction * 1000)

decisionTreeRegressor.fit(x_train, y_train)
prediction = decisionTreeRegressor.predict(x_test)
print(prediction * 1000)

randomForestRegressor.fit(x_train, y_train)
prediction = randomForestRegressor.predict(x_test)
print(prediction * 1000)

supportVectorRegressor.fit(x_train, y_train)
prediction = supportVectorRegressor.predict(x_test)
print(prediction * 1000)
