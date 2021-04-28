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

from utils.utils import *
from utils.clarkWestTest import clarkWestTest


def readFile(architecture, dataset, variable, hidden = None):
    if(hidden is not None):
        if(variable == 'ALL'):
            results = pd.read_parquet('output/' + str(architecture) + '_' + str(dataset) +'_' + str(hidden).replace('[', '').replace(']', '').replace(', ', '_') +  '.gzip')
        else:
             results = pd.read_parquet('output/' + str(architecture) + '_' + str(dataset) +'_' + str(hidden).replace('[', '').replace(']', '').replace(', ', '_') + '_' + str(variable).replace(' ', '').replace('%', '') + '.gzip')
    else: 
        if(variable == 'ALL'):
            results = pd.read_parquet('output/' + str(architecture) + '_' + str(dataset) + '.gzip')
        else:
            results = pd.read_parquet('output/' + str(architecture) + '_' + str(dataset) +'_' + str(variable).replace(' ', '').replace('%', '') + '.gzip')
            
    return results


# # Amalagamation
# The main idea is to do an amalgamation per variable: i.e. the predictions of all architecture at time point t and take the average of those. This begs the question of how to combine them. For example LSTM1/CNN1 does not have the same structure as FNN1. 
#
# Amalgamtion of the all models should be doable, things get more complicated once we start looking at all the variables.
#
# * Amalagamation MEV
# * Amalgamation TA
# * Amalgamtion ALL 
# * Amalgamation PCA MEV
# * Amalgamation PCA TA
# * Amalgamation PCA ALL
# * Amalgamtion per MEV variable 
# * Amalgamation per TA variable 

# +
def amalgamateResults(results, architectures, dataset, PCA = False): 
    """
    The main idea is that I try to aggregate the predictions over all architectures with all hidden units for a single variable. Thus aggregatedict is a dictionairy with dataframe of all the predictions per varaible.
    Thus aggregatedDict{'DP'} would yield a dataframe where each column is the prediction vector of a architecture. Aka, when you take the row wise average you have an amalgamation for a variable based on all models. 
    """
    aggregatedDict = dict()
    variables = results.Dataset.unique()
    models = results.Method.unique()

    
    for architecture in architectures:
        # Set the hidden unit definition depending on the architecture
        if(architecture == 'CNN' or architecture == 'RF'):
            hidden_sizes = None
        elif(architecture == 'MLP' or architecture == 'LSTM'):
            hidden_sizes = [32, 16, 8, 4, 2]
        else:
            hidden_sizes = [[32], [32, 16], [32, 16, 8], [32, 16, 8, 4], [32, 16, 8, 4, 2]] 

        for variable in variables:
            variable = variable.replace(dataset+': ', '')
            
            # Edge case for PCA dataset with different naming convention.
            if(PCA == True):
                variable = 'PCA'
            if(hidden_sizes is not None):
                for hidden in hidden_sizes:
                    #If we are dealing with ALL models, then file naming has a different sctructure.
                    if(variable == 'ALL'):
                        results = pd.read_parquet('output/' + str(architecture) + '_' + str(dataset) +'_' + str(hidden).replace('[', '').replace(']', '').replace(', ', '_') +  '.gzip')
                    else:
                        #For each variation of a certain variable read the relevant file and concatenate the predictions to the dataframe stored in the aggrefatedDict dictionairy for said variable.
                        results = pd.read_parquet('output/' + str(architecture) + '_' + str(dataset) +'_' + str(hidden).replace('[', '').replace(']', '').replace(', ', '_') + '_' + str(variable).replace(' ', '').replace('%', '') + '.gzip')
                    
                    try:
                        df = pd.concat([aggregatedDict.get(variable), results.Pred], axis = 1)
                    except:
                        df = results.Pred
                    aggregatedDict.update({variable: df})

            #For the architectures withouth hidden units, those are irrelevant. 
            elif(hidden_sizes is None):
                #If we are dealing with ALL models, then file naming has a different sctructure.
                if(variable == 'ALL'):
                    results = pd.read_parquet('output/' + str(architecture) + '_' + str(dataset) + '.gzip')
                else:
                    results = pd.read_parquet('output/' + str(architecture) + '_' + str(dataset) +'_' + str(variable).replace(' ', '').replace('%', '') + '.gzip')
                    
                try:
                    df = pd.concat([aggregatedDict.get(variable), results.Pred], axis = 1)
                except:
                    df = results.Pred
                aggregatedDict.update({variable: df})
                
    return aggregatedDict
            
   
# -

def getAmalgamationResults(results, aggregatedDict, dataset, PCA = False):
    resultsDF = pd.DataFrame(columns=['Method', 'Dataset', 'R2', 'CW', 'DA', 'DA HA', 'MSFE', 'MSFE HA'])
    variables = results.Dataset.unique()
    models = results.Method.unique()
    
    # Get the amalgamation performance for each variable
    for variable in variables:
        variable = variable.replace(dataset+': ', '')
        
        if(PCA == True):
            variable = 'PCA'
            
        #Get the amalgamated (row wise average over all model predictions) prediction for a variable
        pred = aggregatedDict.get(variable).mean(axis=1)

        #Get the actual and HA from any file, they are identical in all. 
        results = readFile('MLP', dataset , str(variable), [32])

        # Replace the predictions int he dataframe with the amalgamated predictions
        results.Pred = pred
        
        if(variable is not 'ALL'):
            # Edge case for PCA dataset with different naming convention.
            
            resultsDF = analyzeResults(results, resultsDF, 'Amalgamation', dataset + ': ' + str(variable))
        else: 
            resultsDF = analyzeResults(results, resultsDF, 'Amalgamation', str(variable))
    
    return resultsDF



# ### Amalgamation per variable MEV

# +
# Variables setup: 
results = pd.read_excel(open('output/ALL.xlsx', 'rb'), sheet_name='MEV Variables', engine='openpyxl', index_col=0)
architectures = ['CNN', 'MLP', 'FNN']
dataset = 'MEV'

# Get amalgamated predictions
aggregatedDict = amalgamateResults(results, architectures, dataset)

# Get results based on amalgamation
resultsMEV = getAmalgamationResults(results, aggregatedDict, dataset = 'MEV')
resultsMEV
# -

# ### Amalgamation per variable TA

# +
# Variables setup: 
results = pd.read_excel(open('output/ALL.xlsx', 'rb'), sheet_name='TA Variables', engine='openpyxl', index_col=0)
architectures = ['CNN', 'MLP', 'FNN']
dataset = 'TA'

# Get amalgamated predictions
aggregatedDict = amalgamateResults(results, architectures, dataset)

# Get results based on amalgamation
resultsTA = getAmalgamationResults(results, aggregatedDict, dataset = 'TA')
resultsTA
# -

# ### Amalgamation for ALL model (MEV + TA)

# +
# Variables setup: 
results = pd.read_excel(open('output/ALL.xlsx', 'rb'), sheet_name='Accuracy All', engine='openpyxl', index_col=0)
architectures = ['CNN', 'MLP', 'FNN']
dataset = 'ALL'

# Get amalgamated predictions
aggregatedDict = amalgamateResults(results, architectures, dataset)

# Get results based on amalgamation
resultsALL = getAmalgamationResults(results, aggregatedDict, dataset = 'ALL')
resultsALL
# -

# ### Amalgamation for PCA model (MEV, TA, MEV + TA)

# +
results = pd.read_excel(open('output/ALL.xlsx', 'rb'), sheet_name='Accuracy PCA', engine='openpyxl', index_col=0)
results = results[results.Dataset == 'MEV']
architectures = ['CNN', 'MLP', 'FNN']

# Get amalgamated predictions
aggregatedDict = amalgamateResults(results, architectures, 'MEV', PCA = True)

# Get results based on amalgamation
resultsPCA = getAmalgamationResults(results, aggregatedDict, dataset = 'MEV', PCA = True)

# Redo analysis for TA only PCA models
results = pd.read_excel(open('output/ALL.xlsx', 'rb'), sheet_name='Accuracy PCA', engine='openpyxl', index_col=0)
results = results[results.Dataset == 'TA']
aggregatedDict = amalgamateResults(results, architectures, 'TA', PCA = True)
resultsPCA = resultsPCA.append(getAmalgamationResults(results, aggregatedDict, dataset = 'TA', PCA = True))

# Redo analysis for TA+MEV PCA models
results = pd.read_excel(open('output/ALL.xlsx', 'rb'), sheet_name='Accuracy PCA', engine='openpyxl', index_col=0)
results = results[results.Dataset == 'ALL']
aggregatedDict = amalgamateResults(results, architectures, 'ALL', PCA = True)
resultsPCA = resultsPCA.append(getAmalgamationResults(results, aggregatedDict, dataset = 'ALL', PCA = True))

resultsPCA
# -




