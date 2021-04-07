# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import sklearn
import cProfile
import pstats
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
from utils.dm_test import dm_test
from tqdm.notebook import tqdm #This is not a functional neccesity, but improves the interface. 

pd.set_option('display.max_columns', None)


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
#     y = y
    return X, y


def R2(actual, predicted, average):
    SSR = sum((actual-predicted)**2)
    SST = sum((actual-average)**2)
    return (1- SSR/SST)


def significanceLevel(stat, pVal):
    if(pVal < 0.01):
        return str(round(stat,2)) + '***'
    elif(pVal < 0.05):
        return str(round(stat,2)) + '**'
    elif(pVal < 0.1):
        return str(round(stat,2)) + '*'
    else:
        return str(round(stat,2))


# ## Reading Data
# First we start with loading the relevant data from the excel to be used in our analyis

#Read the equity premium series to a dataframe
ep = pd.read_excel('data/Augemented_Formatted_results.xls', sheet_name='Equity premium', skiprows= range(1118,1127,1))[:-1]
ep['Date'] = pd.to_datetime(ep['Date'], format='%Y%m')
ep = ep.set_index('Date')
ep = ep.loc[(ep.index >= '1950-12-01')]

#Read the maacroeconomic variables to a dataframe
mev = pd.read_excel('data/Augemented_Formatted_results.xls', sheet_name='Macroeconomic variables', 
                    skiprows= range(1118,1126,1)).fillna(method='bfill')[:-1] #backward fill missing values. 
mev = mev.loc[:, ~mev.columns.str.match('Unnamed')]  #Remove empty column
mev['Date'] = pd.to_datetime(mev['Date'], format='%Y%m') #convert date pandas format
mev = mev.set_index('Date') #Set date as index. 
mev = mev.loc[(mev.index >= '1950-12-01')]

ta = pd.read_excel('data/Augemented_Formatted_results.xls', sheet_name='Technical indicators', 
                    skiprows= range(1118,1119,1))[:-1]
ta['Date'] = pd.to_datetime(ta['Date'], format='%Y%m')
ta = ta.set_index('Date')
ta = ta.loc[(ta.index >= '1950-12-01')]

# # Random Forest
# In the code below we will train a random forest for each macro economic variable and technical indicator separately using a rolling window of the past 12 months for each variable/indicator. The recorded R2 score is based on in sample analysis, but the MAE, MSE and MSE are calculated using out of sample analysis. Thus the random forest is trained ussing rolling windows from 1950:12 to 1965:12 yielding 180-12=168 rolling windows. The model is then assessed in terms of prediction accuracy in MAE, MSE and RMSE using data from 1966:01 to 2019:12 yielding 648 rolling windows. 
# ### Macro Economic Variables

window_size = 180
resultsRF = pd.DataFrame(columns=['Method', 'Dataset', 'R2', 'DM']) 

#Shift y variable by 1 month to the future and remove the last observation of the independent variables. Now X and y line up such that each row in X is at time t
# and each row in y with the same index is t+1.
y_mev = ep.shift(periods=-1)[:ep.shape[0]-1].reset_index(drop=True)['Log equity premium'].astype('float64')
X_mev = mev.iloc[:mev.shape[0]-1]


def trainRandomForest(X_mev, y_mev, window_size):
    results_mev = pd.DataFrame(columns=['Date_t', 'Date_t1', 'Actual', 'Pred', 'HA']) 

    for i in tqdm(range(0, X_mev.shape[0]-window_size)):
        #Slice the 180 month rolling window to train the model
        X = X_mev.iloc[i:(window_size + i):]
        y = y_mev[i:(window_size + i):]

        #Get the X and y datapoint at time t (the most recent one) and seperate from the training set. 
        X_t = X.tail(1)
        y_t1 = y.tail(1)
        X = X.iloc[:X.shape[0]-1]
        y = y.iloc[:y.shape[0]-1]

        #Train a random forest model on current slice of data
        RF = RandomForestRegressor(n_estimators = 300, max_depth = 6, random_state = 42).fit(X, y)

        #Make a 1 month OOS prediction of the current time point t.
        y_pred = RF.predict(X_t)

        #Calculate the historical average based on all returns in the current window
        HA = y.mean()

        results_mev = results_mev.append(pd.Series({
            'Date_t': X_t.index.format()[0],
            'Date_t1': ep.index[window_size+i],
            'Actual': y_t1.values.astype('float64')[0],
            'Pred': y_pred[0],
            'HA': HA
        }), ignore_index=True)
        
    return results_mev


# + jupyter={"source_hidden": true}
# profile = cProfile.Profile()
# profile.enable()
# results_mev = trainRandomForest(X_mev, y_mev, window_size)
# profile.disable()
# ps = pstats.Stats(profile)
# ps.sort_stats('cumulative').print_stats(0.1)
# -

results_mev = trainRandomForest(X_mev, y_mev, window_size)

DM = dm_test(results_mev['Actual'].astype(float), results_mev['HA'].astype(float), results_mev['Pred'].astype(float))
resultsRF = resultsRF.append(pd.Series({
            'Method': 'Random Forest',
            'Dataset': 'MEV',
            'R2': round(R2(results_mev.Actual, results_mev.Pred, results_mev.HA) , 3),
            'DM': significanceLevel(DM[0], DM[1])
        }), ignore_index=True)

# ### Technical Indiciators
#

#Shift y variable by 1 month to the future and remove the last observation of the independent variables. Now X and y line up such that each row in X is at time t
# and each row in y with the same index is t+1.
y_ta = ep.shift(periods=-1)[:ep.shape[0]-1].reset_index(drop=True)['Log equity premium'].astype('float64')
X_ta = ta.iloc[:ta.shape[0]-1]

results_ta = trainRandomForest(X_ta, y_ta, window_size)

DM = dm_test(results_ta['Actual'].astype(float), results_ta['HA'].astype(float), results_ta['Pred'].astype(float))
resultsRF = resultsRF.append(pd.Series({
            'Method': 'Random Forest',
            'Dataset': 'TA',
            'R2': round(R2(results_ta.Actual, results_ta.Pred, results_ta.HA) , 3),
            'DM': significanceLevel(DM[0], DM[1])
        }), ignore_index=True)

# In the result below the follow elements can be found:
# * R2 = The out of sample $R^2$ score as defined by eq. 25 in the thesis. 
# * DM: The test statistic for a two-sided Diebold Mariano test with its significance level. The null hypothesis is that there is no difference between the forecasts. When the result is significant there is a difference: posistive values for the DM statistic indicate that the model outperforms the historical average while negative values indicate thet the historical average performs better than the model.
#     * $H_0$: No statisstically significant difference between the forecasts of the model and the historical average
#     * $H_A$: Forcasts of model are significantly difference from historical average. 
#         * Negative DM: Model is significantly worse than HA
#         * Positive DM: Model is signifcantly better than HA

resultsRF

with pd.ExcelWriter('output/RandomForest.xlsx') as writer:
    resultsRF.to_excel(writer, sheet_name='Accuracy')

# # Principal Components Analysis


