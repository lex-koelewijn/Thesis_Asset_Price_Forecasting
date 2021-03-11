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

import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tqdm.notebook import tqdm

pd.set_option('display.max_columns', None)


def createRollingWindow(dataset, look_back = 1):
    """
    Function takes a 2 dimensional array as input and outputs a 2 dimensional array containing rolling windows of the matrix of size [No_Obs - look_back, look_back * No_Vars].
    It creates rolling windows through concatenating all variables at time t with all variables at time t+1 etc until you you have reached t+look_back and move to next window. 
    """
    X= pd.DataFrame(np.empty((dataset.shape[0] - look_back, dataset.shape[1] * look_back)))
    for i in tqdm(range(dataset.shape[0] - look_back)):    
        X.iloc[i] = dataset.iloc[i:(i + look_back):].to_numpy().flatten()
    return X


def createRollingWindow1D(dataset, look_back = 1):
    """
    Function takes a 1 dimensional array as input and outputs a 2 dimensional array containing rolling windows of the series of size look_back.
    """
    X= pd.DataFrame(np.empty((dataset.shape[0] - look_back, look_back)))
    for i in tqdm(range(dataset.shape[0] - look_back)):    
        X.iloc[i] = dataset.iloc[i:(i + look_back):].to_numpy().flatten()
    return X


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

# +
#Shift equity premiumms such that they correspond to the 1 month out of sample corresponding to each window. 
y = ep.shift(periods=-12)[:ep.shape[0]-12].reset_index(drop=True)

#Convert y to a series with only log equity premium or simple equity premium 
y = y['Log equity premium'].astype('float64')

# +
# Create empty dictionary
rollingWindowsMEV = dict()

#Fill the dictionairy with the 2D array with rolling windows for each variable. 
for variable in mev.columns:
    rollingWindowsMEV[variable] = createRollingWindow1D(mev[variable], 12)

# +
df = pd.DataFrame(columns=['Variable', 'R2', 'MAE', 'MSE', 'RMSE'])

for variable in mev.columns:
    X_train, X_test, y_train, y_test = train_test_split(rollingWindowsMEV[variable], y, train_size=168, random_state=0, shuffle=False)
    reg = RandomForestRegressor(random_state = 42).fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    df = df.append(pd.Series({'Variable' : variable, 
                              'R2':  reg.score(X_train, y_train), 
                              'MAE': metrics.mean_absolute_error(y_test, y_pred),
                              'MSE': metrics.mean_squared_error(y_test, y_pred), 
                              'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_pred))}), ignore_index=True)
# -

df

# ### Technical Indiciators
#

# +
#Shift equity premiumms such that they correspond to the 1 month out of sample corresponding to each window. 
y = ep.shift(periods=-12)[:ep.shape[0]-12].reset_index(drop=True)

#Convert y to a series with only log equity premium or simple equity premium 
y = y['Log equity premium'].astype('float64')

# +
# Create empty dictionary
rollingWindowsTA = dict()

#Fill the dictionairy with the 2D array with rolling windows for each variable. 
for variable in ta.columns:
    rollingWindowsTA[variable] = createRollingWindow1D(ta[variable], 12)

# +
df = pd.DataFrame(columns=['Variable', 'R2', 'MAE', 'MSE', 'RMSE'])

for variable in ta.columns:
    X_train, X_test, y_train, y_test = train_test_split(rollingWindowsTA[variable], y, train_size=168, random_state=0, shuffle=False)
    reg = RandomForestRegressor(random_state = 42).fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    df = df.append(pd.Series({'Variable' : variable, 
                              'R2':  reg.score(X_train, y_train), 
                              'MAE': metrics.mean_absolute_error(y_test, y_pred),
                              'MSE': metrics.mean_squared_error(y_test, y_pred), 
                              'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_pred))}), ignore_index=True)
# -

df






