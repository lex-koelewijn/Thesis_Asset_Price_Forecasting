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
from tqdm.notebook import tqdm

pd.set_option('display.max_columns', None)


def createRollingWindow(dataset, look_back=1):
    X= pd.DataFrame(np.empty((dataset.shape[0]-look_back, dataset.shape[1]*look_back)))
    for i in tqdm(range(dataset.shape[0]-look_back)):    
        X.iloc[i] = dataset.iloc[i:(i+look_back):].to_numpy().flatten()
    return X


def shift_data(steps, X, y):
    X = X[:X.shape[0]-steps]
    y = y.shift(periods=-steps)[:y.shape[0]-steps].reset_index(drop=True)
    return X,y


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

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2:', metrics.r2_score(y_test, y_pred))
print('Explained Variance:', metrics.explained_variance_score(y_test, y_pred))
# -

# ### Train OLS Model Vanilla
# Train an OLS model without the rolling window variation. Here we just shift the equity premium by 1 such that we alling 1 row of MEV measurements with the 1 month out of sample equity premium. 

X = mev[:mev.shape[0]-1]
y = ep['Log equity premium'].shift(periods=-1)[:ep['Log equity premium'].shape[0]-1].reset_index(drop=True)

#Create Train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=168, random_state=0, shuffle=False)

#Train a linear regression model on MEV rolling window data and the corresponding 1 month out of sample equity premium. 
reg = LinearRegression().fit(X_train, y_train)
coefficients = reg.coef_
intercept = reg.intercept_

# +
#Make a prediction
y_pred = reg.predict(X_test)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2:', metrics.r2_score(y_test, y_pred))
print('Explained Variance:', metrics.explained_variance_score(y_test, y_pred))
# -

# ## WIP Notes
# * What type of OLS regression should I run? Based on MEV and TA seperately I suppose? Perhaps read rapach 
# * Right now I have contatenated all MEV into one big regression, I'm fairly sure I should do each regression seperately per variable.



mev


