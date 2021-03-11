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

# # OLS Baseline 
# In this notebook an OLS estimation will be performed which serves as a baseline to compare the other methods with. It also serves as the proof-of-concept to figure out how to structure the data correctly and save results before moving on to other architectures.

import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
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

# # Comparisson to Rapach
# I start by recreating the analysis as done in the Neely, Rapach, Tu and Zhou (2014) paper as a starting reference point. Seeing as the results do line up with those presented in the paper, we can confirm that the dataset is the same and all is in order. 
#
# It is the exact in-sample predicitve regression as run in the Rapach paper of which the results can be found in table 2. It is the following bi-variate regression which is run on the data from 1951:01 - 2011:12\:  
# $$ r_{t+1} = \alpha_i +\beta_i q_{i,t} + \epsilon_{i,t+1}$$
#
# First I recreate the MEV regression followed by the TA regressions to make sure the results line up. 
#
# ### MEV

# +
#Shift equity premiumms such that they correspond to the 1 month out of sample corresponding to each window. 
y = ep.shift(periods=-1)[:ep.loc[(ep.index <= '2011-12-01')].shape[0]-1].reset_index(drop=True)

#Convert y to a series with only log equity premium or simple equity premium 
y = y['Log equity premium'].astype('float64')

# Remove the last observation such that the size of the dataamtrix coincides with the shifted y euity ridk premium
X = mev[:mev.loc[(mev.index <= '2011-12-01')].shape[0]-1]

# +
df = pd.DataFrame(columns=['Variable', 'Coef', 'Intercept', 'R2'])
for variable in mev.columns:
#     X_train, X_test, y_train, y_test = train_test_split(, y, train_size=168, random_state=0, shuffle=False)
    reg = LinearRegression().fit(X[variable].values.reshape(-1, 1), y)
    df = df.append(pd.Series({'Variable' : variable, 
                              'Coef' : reg.coef_[0], 
                              'Intercept' : reg.intercept_, 
                              'R2':  reg.score(X[variable].values.reshape(-1,1), y)}), ignore_index=True)

    
# -

df

# ### TA

# Remove the last observation such that the size of the dataamtrix coincides with the shifted y euity ridk premium
X = ta[:ta.loc[(ta.index <= '2011-12-01')].shape[0]-1]

df = pd.DataFrame(columns=['Variable', 'Coef', 'Intercept', 'R2'])
for variable in ta.columns:
#     X_train, X_test, y_train, y_test = train_test_split(, y, train_size=168, random_state=0, shuffle=False)
    reg = LinearRegression().fit(X[variable].values.reshape(-1, 1), y)
    df = df.append(pd.Series({'Variable' : variable, 
                              'Coef' : reg.coef_[0], 
                              'Intercept' : reg.intercept_, 
                              'R2':  reg.score(X[variable].values.reshape(-1,1), y)}), ignore_index=True)

df

# ## Rolling Window Regression
#
# In the analysis below I have implemented multiple versions of a rolling window variation of the regression. 1
# 1. First we create a rolling window of all the MEVs of the past 12 months and concatenate them into 1 vector which serves as input for the 1 month out of sample log equity risk premium. 
# 2. Secondly we create a rolling window of each MEV separately of the past 12 months and concatenate them into 1 vector which serves as input for the 1 month out of sample log equity risk premium. The difference with the first regression is thus that this in this regression we run the analysis for each variable separately to be more in line with Neely, Rapach, Tu and Zhou (2014)/

# ### Data restructuring
# We must create rolling windows of the Macro Economic Variables (MEV) and match them with the 1 month out of sample equity premium in order train a model. 

#Create rolling window version of the MEV dataset.  
X_mev = createRollingWindow(mev, look_back = 12)

# +
#Shift equity premiumms such that they correspond to the 1 month out of sample corresponding to each window. 
y = ep.shift(periods=-12)[:ep.shape[0]-12].reset_index(drop=True)

#Convert y to a series with only log equity premium or simple equity premium 
y = y['Log equity premium'].astype('float64')
# -

# ### Train OLS Model Windowed
# Create rolling windows where we try to predit the 1 month out of sample equity premium based on the previous 12 months of Macro economic variables.

#Create Train and test set
X_train, X_test, y_train, y_test = train_test_split(X_mev, y, train_size=168, random_state=0, shuffle=False)

#Train a linear regression model on MEV rolling window data and the corresponding 1 month out of sample equity premium. 
reg = LinearRegression().fit(X_train, y_train)
coefficients = reg.coef_
intercept = reg.intercept_

# +
#Make a prediction
y_pred = reg.predict(X_test)

print('Coefficients: ', coefficients)
print('Intercept: ', intercept)


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2:', metrics.r2_score(y_test, y_pred))
print('Explained Variance:', metrics.explained_variance_score(y_test, y_pred))
# -

# ## Rolling window OLS estimation for each variable separately
# Up till now I have combined all variables into 1 window, however this is not in line with Neely, Rapach, Tu and Zhou (2014). They run seperate regression for each macro economic variable and TA variable, and hence that is what should also do. 
#
# rollingWindowMEV is a dictionary which will be filled with the 2D dataframes containing the rolling windows for a single variable. Thus rollingWindowsMEV['DP'] would yield the 2D matrix containing the rolling windows for the variable DP and will be of size [No_obs-window_size, window_size] or in our case [828 obs - 12, 12] = [817,12]. The dictionairy is thus accessible with the variable name as index key and these can be obtained from the original data through mev.columns which yields an array of the column names of the original dataframe.

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
df = pd.DataFrame(columns=['Variable', 'Coef', 'Intercept', 'R2', 'MAE', 'MSE', 'RMSE'])

for variable in mev.columns:
    X_train, X_test, y_train, y_test = train_test_split(rollingWindowsMEV[variable], y, train_size=168, random_state=0, shuffle=False)
    reg = LinearRegression().fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    df = df.append(pd.Series({'Variable' : variable, 
                              'Coef' : reg.coef_[0], 
                              'Intercept' : reg.intercept_, 
                              'R2':  reg.score(X_train, y_train),
                              'MAE': metrics.mean_absolute_error(y_test, y_pred),
                              'MSE': metrics.mean_squared_error(y_test, y_pred), 
                              'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_pred))}), ignore_index=True)
# -

df

# +
# Create empty dictionary
rollingWindowsTA = dict()

#Fill the dictionairy with the 2D array with rolling windows for each variable. 
for variable in ta.columns:
    rollingWindowsTA[variable] = createRollingWindow1D(ta[variable], 12)

# +
df = pd.DataFrame(columns=['Variable', 'Coef', 'Intercept', 'R2', 'MAE', 'MSE', 'RMSE'])

for variable in ta.columns:
    X_train, X_test, y_train, y_test = train_test_split(rollingWindowsTA[variable], y, train_size=168, random_state=0, shuffle=False)
    reg = LinearRegression().fit(X_train, y_train)
    y_pred = reg.predict(X_test)    
    df = df.append(pd.Series({'Variable' : variable, 
                              'Coef' : reg.coef_[0], 
                              'Intercept' : reg.intercept_, 
                              'R2':  reg.score(X_train, y_train),
                              'MAE': metrics.mean_absolute_error(y_test, y_pred),
                              'MSE': metrics.mean_squared_error(y_test, y_pred), 
                              'RMSE': np.sqrt(metrics.mean_squared_error(y_test, y_pred))}), ignore_index=True)
# -

df


