import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

def getGridSearchRMSE(_param):
    #tune GridSearch with _param
    gridSearch = GridSearchCV(estimator=randomForestRegressor, 
        param_grid=_param, cv=10)
    
    #compute RMSE
    gridSearch.fit(x_train, y_train)
    prediction = gridSearch.predict(x_test)
    err = prediction - y_test
    x_err = 0
    x_err += np.dot(err, err)
    rmse = np.sqrt(x_err/len(x))
    return rmse

def printGridSearchResults():
    #assignment parameters
    param_grid = {
        'n_estimators': [10,20,30,40],
        'max_features': [4,8]
    }

    meanRMSE = getGridSearchRMSE(param_grid)

    print('Grid Search\tMean RMSE\t', meanRMSE)
    
    #found through using best_params_ on the above model
    final_model = {
        'n_estimators': [40],
        'max_features': [4]
    }

    print('Best Parameters:', final_model)
    bestRMSE = getGridSearchRMSE(final_model)
    print('Best Grid Search\tRMSE\t', bestRMSE)

def doKFold(_name, _model):
    x_err = 0
    iteration = 1
    predictionSum = 0
    deviationSum = 0

    for train, test in kFold.split(x):
        _model.fit(x.iloc[train], y[train])
        prediction = _model.predict(x.iloc[test])
        
        #Needed to calculate average mean and standard deviation
        predictionSum += np.mean(prediction)
        deviationSum += np.std(prediction)
        
        err = prediction - y[test]
        x_err += np.dot(err, err)
        rmse = np.sqrt(x_err/len(x))
        
        print(_name, '\n\tRMSE Fold #', iteration, '\t', rmse)
        iteration += 1

    meanPrediction = predictionSum/10
    meanDeviation = deviationSum/10

    print('Mean Prediction:', meanPrediction, 'Average Deviation:', meanDeviation, '\n')

if __name__ == '__main__':
    #initialize all of the models
    standardScaler = StandardScaler()
    linearRegression = LinearRegression()
    decisionTreeRegressor = DecisionTreeRegressor()
    randomForestRegressor = RandomForestRegressor()
    supportVectorRegressor = SVR(kernel='rbf', gamma=0.01)
    kFold = KFold(n_splits=10, random_state=7)

    #load data
    data = pd.read_csv('data/boston_housing.data', header=None, delimiter=' ')

    #dataframe to reduce repetition of code
    modelDataframe = pd.DataFrame({
        'name': ['LinearRegression', 'DecisionTreeRegressor',
        'RandomForestRegressor', 'SupportVectorRegressor'], 
        'model': [linearRegression, decisionTreeRegressor,
        randomForestRegressor, supportVectorRegressor]
    })

    #x is all of the features except the last column
    x = data[range(13)]
    #y is our target, per the assignment, we are to multiply it by 1000
    y = data[13] * 1000

    #split the test/train data 30%/70%
    x_train, x_test, y_train, y_test = train_test_split(x, y, 
        test_size=0.3, random_state=7)

    #scale both training data and testing data
    x_train = standardScaler.fit_transform(x_train)
    x_test = standardScaler.fit_transform(x_test)

    #loops through each model in the dataframe
    for index, row in modelDataframe.iterrows():
        doKFold(row['name'], row['model'])
        
    printGridSearchResults()
