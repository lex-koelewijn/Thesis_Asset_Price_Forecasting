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


def createRollingWindow1D(dataset, look_back = 1):
    """
    Function takes a 1 dimensional array as input and outputs a 2 dimensional array containing rolling windows of the series of size look_back. (Where each row is a rolling window)
    """
    X= pd.DataFrame(np.empty((dataset.shape[0] - look_back, look_back)))
    for i in range(dataset.shape[0] - look_back):    
        X.iloc[i] = dataset.iloc[i:(i + look_back):].to_numpy().flatten()
    return X


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
#Initialize empty dataframe
df_mev = pd.DataFrame(columns=['Variable', 'R2 IS', 'R2 OOS','R2 OOS HA', 'DM', 'MAE', 'MAE HA', 'MSE', 'MSE HA', 'RMSE', 'RMSE HA'])
y_pred = []

for variable in mev.columns:
    #Split data into appropriate train and test size
    X_train, X_test, y_train, y_test = train_test_split(rollingWindowsMEV[variable], y, train_size=168, random_state=0, shuffle=False) #Number of training exmples is defined here, move to global. 
    
    #Create the OOS historical average benchmark
    HA = createRollingWindow1D(ep['Log equity premium'].astype('float64'), 12)
    ha_test = HA.mean(axis=1).loc[168:]

    #Define and train the model and evaluate OOS performance. 
    reg = RandomForestRegressor(random_state = 42).fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    
    #Compare current model's predictions to the historical average benchmark through Diebold-Mariano test
    DM = dm_test(y_test.astype(float), ha_test.astype(float), pd.Series(y_pred).astype(float))
    
    df_mev = df_mev.append(pd.Series({'Variable' : variable, 
                              'R2 IS':  reg.score(X_train, y_train), 
                              'R2 OOS':  metrics.r2_score(y_test, y_pred),
                              'R2 OOS HA': metrics.r2_score(y_test, ha_test),       
                              'DM':  significanceLevel(DM[0], DM[1]),       
                              'MAE': metrics.mean_absolute_error(y_test, y_pred),
                              'MAE HA': metrics.mean_absolute_error(y_test, ha_test),
                              'MSE': metrics.mean_squared_error(y_test, y_pred),
                              'MSE HA': metrics.mean_squared_error(y_test, ha_test),
                              'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
                              'RMSE HA': np.sqrt(metrics.mean_squared_error(y_test, ha_test))}) , ignore_index=True)
# -

# In the result below the follow elements can be found:
# * R2 IS = The in sample $R^2$ score, aka the $R^2$ achieved by the model in the train set
# * R2 OOS = The out of sample $R^2$ score, aka the $R^2$ achieved by the model on the test set
# * R2 OOS Ha = The out of sample $R^2$ score achieved by the historical average
# * DM: The test statistic for the Diebold Mariano test with its significance level. The null hypothesis is that there is no difference between the forecasts. When the result is significant there is a difference: posistive values for the DM statistic indicate that the model outperforms the historical average while negative values indicate thet the historical average performs better than the model. 
# * MAE: Mean absolute error of prediction and true value for model and HA.
# * MSE: Mean sqaured error of prediction and true value for model and HA.
# * RMSE: Root mean squared error of prediction and true value for the model and HA. 

df_mev.round(4)

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
#Initialize empty dataframe
df_ta = pd.DataFrame(columns=['Variable', 'R2 IS', 'R2 OOS','R2 OOS HA', 'DM', 'MAE', 'MAE HA', 'MSE', 'MSE HA', 'RMSE', 'RMSE HA'])
y_pred = []

for variable in ta.columns:
    #Split data into appropriate train and test size
    X_train, X_test, y_train, y_test = train_test_split(rollingWindowsTA[variable], y, train_size=168, random_state=0, shuffle=False) #Number of training exmples is defined here, move to global. 
    
    #Create the OOS historical average benchmark
    HA = createRollingWindow1D(ep['Log equity premium'].astype('float64'), 12)
    ha_test = HA.mean(axis=1).loc[168:]

    #Define and train the model and evaluate OOS performance. 
    reg = RandomForestRegressor(random_state = 42).fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    
    #Compare current model's predictions to the historical average benchmark through Diebold-Mariano test
    DM = dm_test(y_test.astype(float), ha_test.astype(float), pd.Series(y_pred).astype(float))
    
    df_ta = df_ta.append(pd.Series({'Variable' : variable, 
                              'R2 IS':  reg.score(X_train, y_train), 
                              'R2 OOS':  metrics.r2_score(y_test, y_pred),
                              'R2 OOS HA': metrics.r2_score(y_test, ha_test),       
                              'DM':  significanceLevel(DM[0], DM[1]),       
                              'MAE': metrics.mean_absolute_error(y_test, y_pred),
                              'MAE HA': metrics.mean_absolute_error(y_test, ha_test),
                              'MSE': metrics.mean_squared_error(y_test, y_pred),
                              'MSE HA': metrics.mean_squared_error(y_test, ha_test),
                              'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_pred)),
                              'RMSE HA': np.sqrt(metrics.mean_squared_error(y_test, ha_test))}) , ignore_index=True)
# -

df_ta.round(4)

# # Principal Components Analysis




