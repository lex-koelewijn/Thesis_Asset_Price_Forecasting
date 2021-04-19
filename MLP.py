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

# +
import pandas as pd
import numpy as np
import keras 
from utils.utils import *
from utils.clarkWestTest import clarkWestTest

from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
# from tensorflow.python.keras.layers import Input, Dense, Activation
# from tensorflow.python.keras.models import Sequential
from keras import losses 
from keras import optimizers 
from keras import metrics 
from tqdm.notebook import tqdm
# -

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
mev = mev.drop(columns = ['Risk-free rate','12-month moving sum of earnings'])

ta = pd.read_excel('data/Augemented_Formatted_results.xls', sheet_name='Technical indicators', 
                    skiprows= range(1118,1119,1))[:-1]
ta['Date'] = pd.to_datetime(ta['Date'], format='%Y%m')
ta = ta.set_index('Date')
ta = ta.loc[(ta.index >= '1950-12-01')]


# ## Functions

def normalizeData(X):
    return (X-np.mean(X))/np.std(X)


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


def createModel(hidden):
    model = Sequential()
    model.add(Dense(14, input_shape=(14, ), 
                    activation='relu', 
                    activity_regularizer=regularizers.l1(0.01)))
    model.add(BatchNormalization())
    model.add(Dense(hidden, 
                    activation='relu', 
                    activity_regularizer=regularizers.l1(0.01)))
    model.add(BatchNormalization())
    model.add(Dense(1, 
                    activation='linear', 
                    activity_regularizer=regularizers.l1(0.01)))

    model.compile(optimizer = 'sgd', loss = 'mean_squared_error', metrics = ['mean_squared_error'])
    early_stopping = EarlyStopping(monitor='val_loss', patience = 5, min_delta=0.001, mode = 'min')
    model.fit(X_mev, y_mev, epochs=100, batch_size=256, validation_split = 0.2, callbacks=[early_stopping], verbose=0)
    return model


def trainMLP(X_mev, y_mev, window_size, hidden):
    results = pd.DataFrame(columns=['Date_t', 'Date_t1', 'Actual', 'Pred', 'HA']) 
    
    for i in tqdm(range(0, X_mev.shape[0]-window_size)):
        #Slice the 180 month rolling window to train the model
        X = X_mev.iloc[i:(window_size + i):]
        y = y_mev[i:(window_size + i):]

        #Get the X and y datapoint at time t (the most recent one) and seperate from the training set. 
        X_t = X.tail(1)
        y_t1 = y.tail(1)
        X = X.iloc[:X.shape[0]-1]
        y = y.iloc[:y.shape[0]-1]
        
         # Define and train the model in keras once every year
        if(i % 12 == 0):
            model = createModel(hidden)

        #Make a 1 month OOS prediction of the current time point t.
        y_pred = model.predict(X_t)

        #Calculate the historical average based on all returns in the current window
        HA = y.mean()

        results = results.append(pd.Series({
            'Date_t': X_t.index.format()[0],
            'Date_t1': ep.index[window_size+i],
            'Actual': y_t1.values.astype('float64')[0],
            'Pred': y_pred[0][0],
            'HA': HA
        }), ignore_index=True)
        
    return results


def modelTrainingSequence(X, y, window_size, hidden, dataset):
    performanceResults = pd.DataFrame(columns=['Method', 'Dataset', 'R2', 'CW']) 
    
    for hidden in hidden_sizes:
        try: 
            results = pd.read_parquet('output/MLP_' + str(dataset) +'_' + str(hidden) +'.gzip')
        except:
            print('No saved results found, running model estimation.')
            results = trainMLP(X, y, window_size = window_size, hidden = hidden)
            results.to_parquet('output/MLP_' + str(dataset) +'_' + str(hidden) +'.gzip', compression='gzip')
        performanceResults = analyzeResults(results, performanceResults, method = 'MLP '+str(hidden), dataset = dataset)
    
    return performanceResults
    


# ### MLP MEV

window_size = 180
hidden_sizes = [32, 16, 8, 4, 2] 
check_existence_directory(['output'])

y_mev = ep.shift(periods=-1)[:ep.shape[0]-1].reset_index(drop=True)['Log equity premium'].astype('float64')
X_mev = mev.iloc[:mev.shape[0]-1]

resultsMLP = modelTrainingSequence(normalizeData(X_mev), y_mev, window_size, hidden_sizes, dataset = 'MEV')

resultsMLP

# ### resultsMLP

# ### MLP TA

window_size = 180
hidden_sizes = [32, 16, 8, 4, 2] 
check_existence_directory(['output'])

#Shift y variable by 1 month to the future and remove the last observation of the independent variables. Now X and y line up such that each row in X is at time t
# and each row in y with the same index is t+1.
y_ta = ep.shift(periods=-1)[:ep.shape[0]-1].reset_index(drop=True)['Log equity premium'].astype('float64')
X_ta = ta.iloc[:ta.shape[0]-1]

resultsTA = modelTrainingSequence(normalizeData(X_ta), y_ta, window_size, hidden_sizes, dataset = 'TA')
























