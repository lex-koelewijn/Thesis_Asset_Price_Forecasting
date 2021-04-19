import os
import pandas as pd
import numpy as np

def createRollingWindow(dataset, look_back = 1):
    """
    Function takes a 2 dimensional array as input and outputs a 2 dimensional array containing rolling windows of the matrix of size [No_Obs - look_back, look_back * No_Vars].
    It creates rolling windows through concatenating all variables at time t with all variables at time t+1 etc until you you have reached t+look_back and move to next window. 
    """
    X= pd.DataFrame(np.empty((dataset.shape[0] - look_back, dataset.shape[1] * look_back)))
    for i in range(dataset.shape[0] - look_back):    
        X.iloc[i] = dataset.iloc[i:(i + look_back):].to_numpy().flatten()
    return X


def createRollingWindow1D(dataset, returns, look_back = 1):
    """
    Function takes a 1 dimensional array as input and outputs a 2 dimensional array containing rolling windows of the series of size look_back. (Where each row is a rolling window)
    The corresponding returns (y) will also be shifted to be in line with the look_back variable such that the rolling windows and the 1 month OOS return line up.
    """
    X= pd.DataFrame(np.empty((dataset.shape[0] - look_back, look_back)))
    for i in range(dataset.shape[0] - look_back):    
        X.iloc[i] = dataset.iloc[i:(i + look_back):].to_numpy().flatten()
        
    y = returns.shift(periods=-look_back)[:returns.shape[0]-look_back].reset_index(drop=True)['Log equity premium'].astype('float64')
    return X, y


def R2(actual, predicted, average):
    """
    Calculated the R2 score of the vectors containing the actual returns, predicted returns and the average returns. 
    """
    SSR = sum((actual-predicted)**2)
    SST = sum((actual-average)**2)
    return (1- SSR/SST)


def significanceLevel(stat, pVal):
    """
    Function to format statistical test results by adding asterisks for the appropriate significance levels and round the numbers. 
    """
    if(pVal < 0.01):
        return str(round(stat,2)) + '***'
    elif(pVal < 0.05):
        return str(round(stat,2)) + '**'
    elif(pVal < 0.1):
        return str(round(stat,2)) + '*'
    else:
        return str(round(stat,2))    

def directionalAccuracy(actual, predicted):
    """
    Calculate the directional accuracy of a predicted series compared to the actual series. Output is a value between 0 and 1 representing a percentage.
    """
    return round(sum(np.sign(actual) == np.sign(predicted))/actual.shape[0]*100,2)

def check_existence_directory(directories):
    for direc in directories:
        if not os.path.exists(direc):
            os.makedirs(direc)
            
def analyzeResults(results, resultsRF, method, dataset):
    """
    Calcutale the evaluation measures based on the results of a mdel and append them to a datafram provided. 
    """
    CW = clarkWestTest(results['Actual'].astype(float), results['HA'].astype(float), results['Pred'].astype(float))
    resultsRF = resultsRF.append(pd.Series({
                'Method': method,
                'Dataset': dataset,
                'R2': round(R2(results.Actual, results.Pred, results.HA) , 3),
                'CW': significanceLevel(CW[0], CW[1]),
                'DA': directionalAccuracy(results.Actual, results.Pred),
                'DA HA': directionalAccuracy(results.Actual, results.HA)
            }), ignore_index=True)
    return resultsRF