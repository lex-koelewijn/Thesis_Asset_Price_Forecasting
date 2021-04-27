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

results = pd.read_excel(open('output/ALL.xlsx', 'rb'), sheet_name='MEV Variables', engine='openpyxl', index_col=0)
variables = results.Dataset.unique()
models = results.Method.unique()
table = pd.DataFrame(index=variables)
architectures = ['CNN', 'MLP'] #TODO: Incomplete
dataset = 'MEV'
aggregatedDict = dict()

# +
"""
The main idea is that I try to aggregate the predictions over all architectures with all hidden units for a single variable. Thus aggregatedict is a dictionairy with dataframe of all the predictions per varaible.
Thus aggregatedDict{'DP'} would yield a dataframe where each column is the prediction vector of a architecture. Aka, when you take the row wise average you have an amalgamation for a variable based on all models. 
"""
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
        if(hidden_sizes is not None):
            for hidden in hidden_sizes:
                #For each variation of a certain variable read the relevant file and concatenate the predictions to the dataframe stored in the aggrefatedDict dictionairy for said variable.
                results = pd.read_parquet('output/' + str(architecture) + '_' + str(dataset) +'_' + str(hidden).replace('[', '').replace(']', '').replace(', ', '_') + '_' + str(variable).replace(' ', '').replace('%', '') + '.gzip')
                try:
                    df = pd.concat([aggregatedDict.get(variable), results.Pred], axis = 1)
                except:
                    df = results.Pred
                aggregatedDict.update({variable: df})
        
        #For the architectures withouth hidden units, those are irrelevant. 
        elif(hidden_sizes is None):
            results = pd.read_parquet('output/' + str(architecture) + '_' + str(dataset) +'_' + str(variable).replace(' ', '').replace('%', '') + '.gzip')
            try:
                df = pd.concat([aggregatedDict.get(variable), results.Pred], axis = 1)
            except:
                df = results.Pred
            aggregatedDict.update({variable: df})
            
   
# -

aggregatedDict.get('DP')


