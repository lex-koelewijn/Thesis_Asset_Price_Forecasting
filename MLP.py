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

import keras
import tensorflow
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from sklearn.decomposition import PCA
from pandas.testing import assert_series_equal
from keras import optimizers 
from keras import metrics 
from tqdm.notebook import tqdm
from numpy.random import seed


# ## Google Collab Setup 
# The following blocks of code should be used when running the code on Google Collab. (Meaning the local set up block should be commented) 

# +
# import sys
# import os
# import tensorflow as tf
# from google.colab import files
# from google.colab import drive

# +
# # Mount my google drive where the input files are stored. 
# drive.mount('/content/drive')
# sys.path.append('/content/drive/MyDrive/RUG/Master thesis finance/')

# #Import util files from Drive
# from utils.utils import *
# from utils.clarkWestTest import clarkWestTest

# +
# # Load the GPU provided by google
# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#   raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))

# #Set desired verbosity of tensorflow
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# # Read the data from the drive location
# path_to_data = '/content/drive/MyDrive/RUG/Master thesis finance/'
# # Read in the relevant data for the analysis.
# ep = readEquityPremiumData(path_to_data)
# mev = readMacroEconomicVariableData(path_to_data)
# ta = readTechnicalIndicatorData(path_to_data)
# -

def downloadFiles(directory):
    """
    Function which downloads all files in the directory specified in the Collab environment.   
    """
    for filename in os.listdir(directory):
        files.download(directory+filename)


def copyFilesToDrive():
    """
    Function which copies all the output files to my personal drive. Fully hardcoded.   
    """
#     !cp -r 'output/' '/content/drive/MyDrive/RUG/Master thesis finance/'


# ## Local Setup
# The code blocks below should be used when running the repository on a local machine. (Meaning the Google collab block should be commented)

from utils.utils import *
from utils.clarkWestTest import clarkWestTest

ep = readEquityPremiumData()
mev = readMacroEconomicVariableData()
ta = readTechnicalIndicatorData()

# ## Global Setup
# From hereonout all code works both on a local machine or in Google collab. The required settings that are unisversal will be applied below. 

#Set fixed seed for python, numpy and tensorflow
seed(42)
np.random.seed(42)
try:
    tensorflow.random.set_seed(42)
except:
    tensorflow.set_random_seed(42)


# ## Functions

# +
# def normalizeData(X):
#     return (X-np.mean(X))/np.std(X)

# +
# def MSFE(actual, pred):
#     return( np.mean((actual-pred)**2))

# +
# def analyzeResults(results, resultsRF, method, dataset):
#     """
#     Calcutale the evaluation measures based on the results of a mdel and append them to a datafram provided. 
#     """
#     CW = clarkWestTest(results['Actual'].astype(float), results['HA'].astype(float), results['Pred'].astype(float))
#     resultsRF = resultsRF.append(pd.Series({
#                 'Method': method,
#                 'Dataset': dataset,
#                 'R2': round(R2(results.Actual, results.Pred, results.HA) , 3),
#                 'CW': significanceLevel(CW[0], CW[1]),
#                 'DA': directionalAccuracy(results.Actual, results.Pred),
#                 'DA HA': directionalAccuracy(results.Actual, results.HA),
#                 'MSFE': MSFE(results.Actual, results.Pred),
#                 'MSFE HA': MSFE(results.Actual, results.HA)
        
#             }), ignore_index=True)
#     return resultsRF
# -

def createModel(X, y, inputUnits, inputShape, hidden):
    model = Sequential()
    model.add(Dense(inputUnits, input_shape=inputShape, 
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

    model.compile(optimizer = 'ADAM', loss = 'mean_squared_error', metrics = ['mean_squared_error'])
    early_stopping = EarlyStopping(monitor='val_loss', patience = 5, min_delta=0.001, mode = 'min')
    model.fit(X, y, epochs=100, batch_size=8, validation_split = 0.2, callbacks=[early_stopping], verbose=0)
    return model


def trainMLP(X_mev, y_mev, window_size, hidden, inputUnits, inputShape):
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
            model = createModel(X, y, inputUnits, inputShape, hidden)

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


def modelTrainingSequence(X, y, window_size, hidden, architecture, dataset, inputUnits, inputShape):
    performanceResults = pd.DataFrame(columns=['Method', 'Dataset', 'R2', 'CW', 'DA', 'DA HA', 'MSFE', 'MSFE HA'])  
    
    # For each of the network specifications, try to find saved outputs. Otherwise train and evaluate model and save the outcomes. 
    for hidden in hidden_sizes:
        try: 
            results = pd.read_parquet('output/' + str(architecture) + '_' + str(dataset) +'_' + str(hidden) +'.gzip')
        except:
            print('No saved results found, running model estimation.')
            results = trainMLP(X, y, window_size = window_size, hidden = hidden, inputUnits = inputUnits, inputShape = inputShape)
            results.to_parquet('output/' + str(architecture) + '_' + str(dataset) +'_' + str(hidden) +'.gzip', compression='gzip')
        performanceResults = analyzeResults(results, performanceResults, method = str(architecture)+' '+str(hidden), dataset = dataset)
    
    return performanceResults
    


# ### MLP MEV
# Run the MLP for all the macroeconomic at once as training input. 

window_size = 180
hidden_sizes = [32, 16, 8, 4, 2] 
check_existence_directory(['output'])

y_mev = ep.shift(periods=-1)[:ep.shape[0]-1].reset_index(drop=True)['Log equity premium'].astype('float64')
X_mev = mev.iloc[:mev.shape[0]-1]

resultsMEVAll = modelTrainingSequence(normalizeData(X_mev), normalizeData(y_mev), window_size, hidden_sizes, architecture = 'MLP', dataset = 'MEV', inputUnits = 14, inputShape = (14,))

resultsMEVAll

# ### MLP TA
# Run the MLP for all the macroeconomic at once as training input. 

window_size = 180
hidden_sizes = [32, 16, 8, 4, 2] 
check_existence_directory(['output'])

#Shift y variable by 1 month to the future and remove the last observation of the independent variables. Now X and y line up such that each row in X is at time t
# and each row in y with the same index is t+1.
y_ta = ep.shift(periods=-1)[:ep.shape[0]-1].reset_index(drop=True)['Log equity premium'].astype('float64')
X_ta = ta.iloc[:ta.shape[0]-1]

resultsTAAll = modelTrainingSequence(normalizeData(X_ta), normalizeData(y_ta), window_size, hidden_sizes, architecture = 'MLP', dataset = 'TA', inputUnits = 14, inputShape = (14,))

resultsTAAll

# ### MLP ALL
# Run the MLP for all the macroeconomic variables and techical indicators at once. 

assert_series_equal(y_ta, y_mev)
y_all = y_ta
X_all = pd.concat([X_mev, X_ta], ignore_index = False, axis =1)
resultsAll = modelTrainingSequence(normalizeData(X_all), normalizeData(y_all), window_size, hidden_sizes, architecture = 'MLP', dataset = 'ALL', inputUnits = 28, inputShape = (28,))

resultsAll


# # Principal Components Analysis

def trainPCAModel(X, y, explainedVariance, hidden, architecture, dataset):
    performanceResults = pd.DataFrame(columns=['Method', 'Dataset', 'R2', 'CW', 'DA', 'DA HA', 'MSFE', 'MSFE HA']) 
    
    # Let the PCA analysis determine the optimal number of components such that >95% of the variance is explained. 
    pca = PCA(n_components=0.95, svd_solver='full')
    X = pd.DataFrame(pca.fit_transform(normalizeData(X)))
    y = normalizeData(y)
    
    #Extract the input sizes from the PCA transformed dataset. 
    inputUnits = X.shape[1]
    inputShape = (X.shape[1],)
    
    for hidden in hidden_sizes:
        try: 
            results_pca = pd.read_parquet('output/' + str(architecture) + '_' + str(dataset) +'_' + str(hidden)+ '_PCA' +'.gzip')
        except:
            print('No saved results found, running model estimation.')
            results_pca = trainMLP(X, y, window_size = window_size, hidden = hidden, inputUnits = inputUnits, inputShape = inputShape)
            results_pca.to_parquet('output/' + str(architecture) + '_' + str(dataset) +'_' + str(hidden)+ '_PCA' +'.gzip', compression='gzip')
        performanceResults = analyzeResults(results_pca, performanceResults, method = str(architecture)+' PCA '+str(hidden), dataset = dataset)
                                   
    return performanceResults                                



resultsMEVPCA = trainPCAModel(X_mev, y_mev, window_size, hidden_sizes, architecture = 'MLP', dataset = 'MEV')
resultsTAPCA = trainPCAModel(X_ta, y_ta, window_size, hidden_sizes, architecture = 'MLP', dataset = 'TA')
resultsALLPCA = trainPCAModel(X_all, y_all, window_size, hidden_sizes, architecture = 'MLP', dataset = 'ALL')
resultsPCACombined = pd.concat([resultsMEVPCA, resultsTAPCA, resultsALLPCA])


# # Analysis per Variable
# Up till now we have trained models using a vector of all variables at each time point. In the analysis below models will be trained for each variable seperately with the same set up as above. This allows us to observe the predictive power of variables indivdually given the current model architecucture.

def runAnalysisPerVariable(X_raw, y_raw, hidden, window_size, dataset, inputUnits, inputShape):
    # Initialize empty datafram to contain the results. 
    resultsDF = pd.DataFrame(columns=['Method', 'Dataset', 'R2', 'CW']) 
    
    # Init y
    y = y_raw.shift(periods=-1)[:y_raw.shape[0]-1].reset_index(drop=True)['Log equity premium'].astype('float64')

    for variable in X_raw.columns:
        # Select current variable and reshape such that pandas and numpy understand each other. 
        X = X_raw.iloc[:X_raw.shape[0]-1][variable]
        X = pd.DataFrame(X.values.reshape(-1, 1))
        
        for hidden in hidden_sizes:
            # If model has been trained already we load input, otherwise train model. 
            try: 
                results = pd.read_parquet('output/MLP_' + str(dataset) +'_' + str(hidden) + '_' + str(variable).replace(' ', '').replace('%', '') + '.gzip')
            except:
                print('No saved results found, running model estimation.')
                results = trainMLP(normalizeData(X), normalizeData(y), window_size = window_size, hidden = hidden, inputUnits = inputUnits, inputShape = inputShape)
                results.to_parquet('output/MLP_' + str(dataset) +'_' + str(hidden) + '_' + str(variable).replace(' ', '').replace('%', '') + '.gzip', compression='gzip')
                
            #Analyze the results
            resultsDF = analyzeResults(results, resultsDF, method = 'MLP '+str(hidden), dataset =   dataset + ': ' + str(variable))
            
    return resultsDF


# ### Macroeconomic variables

resultsMEV = runAnalysisPerVariable(mev, ep, hidden_sizes,  window_size, dataset = 'MEV', inputUnits = 1, inputShape = (1,))

resultsMEV

# ### Technical Indicators

resultsTA = runAnalysisPerVariable(ta, ep, hidden_sizes,  window_size, dataset = 'TA', inputUnits = 1, inputShape = (1,))

resultsTA

with pd.ExcelWriter('output/MLP.xlsx') as writer:
    resultsMEVAll.to_excel(writer, sheet_name='Accuracy MEV')
    resultsTAAll.to_excel(writer, sheet_name='Accuracy TA')
    resultsAll.to_excel(writer, sheet_name='Accuracy All')
    resultsPCACombined.to_excel(writer, sheet_name='Accuracy PCA')
    resultsMEV.to_excel(writer, sheet_name='MEV Variables')
    resultsTA.to_excel(writer, sheet_name='TA Variables')




