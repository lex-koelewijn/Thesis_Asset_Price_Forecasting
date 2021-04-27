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
import re

files = [ 'MLP.xlsx', 'FNN.xlsx', 'CNN.xlsx', 'LSTM.xlsx', 'RBFNN.xlsx'] #Random forest has different structure rn, add later
direc = 'output/'

resultsMEVAll, resultsTAAll, resultsAll, resultsPCACombined, resultsMEV, resultsTA = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def replaceNames(series):
    """
    Function is fed a numpy array with architecture names and updates them using a dictionairy to be in line with the notation used in the thesis. This new array is returned.  
    """
    # Dict with elements that need to be replaced
    replaceDict = {' [32, 16, 8, 4, 2]': '5', ' [32, 16, 8, 4]': '4', ' [32, 16, 8]': '3', ' [32, 16]': '2', ' [32]': '1'}
    
    # Loop through all the values in the series provided
    for idx, value in enumerate(series):
        # Try to replace each of th elements in the replace dict
        for key in replaceDict.keys():
            value = value.replace(key, replaceDict[key])
        # Update new index value
        series[idx] = value
    return series


# ### Combine all output files to a single one

for file in files:
    resultsMEVAll = resultsMEVAll.append(pd.read_excel(open(direc+file, 'rb'), sheet_name='Accuracy MEV', engine='openpyxl', index_col=0))
    resultsTAAll = resultsTAAll.append(pd.read_excel(open(direc+file, 'rb'), sheet_name='Accuracy TA', engine='openpyxl', index_col=0))
#     resultsAll = resultsAll.append(pd.read_excel(open(direc+file, 'rb'), sheet_name='Accuracy All', engine='openpyxl', index_col=0))
#     resultsPCACombined = resultsPCACombined.append(pd.read_excel(open(direc+file, 'rb'), sheet_name='Accuracy PCA', engine='openpyxl', index_col=0))
    resultsMEV = resultsMEV.append(pd.read_excel(open(direc+file, 'rb'), sheet_name='MEV Variables', engine='openpyxl', index_col=0))
    resultsTA = resultsTA.append(pd.read_excel(open(direc+file, 'rb'), sheet_name='TA Variables', engine='openpyxl', index_col=0))

with pd.ExcelWriter('output/ALL.xlsx') as writer:
    resultsMEVAll.to_excel(writer, sheet_name='Accuracy MEV')
    resultsTAAll.to_excel(writer, sheet_name='Accuracy TA')
    resultsAll.to_excel(writer, sheet_name='Accuracy All')
    resultsPCACombined.to_excel(writer, sheet_name='Accuracy PCA')
    resultsMEV.to_excel(writer, sheet_name='MEV Variables')
    resultsTA.to_excel(writer, sheet_name='TA Variables')


def updateVariableNames(variables, dataset):
    for idx, var in enumerate(variables):
        variables[idx] = var.replace(dataset + ': ', '')
        
    return variables


# # Create a formatted table for the results as we desire it

def createResultsTable(results, dataset):
    variables = results.Dataset.unique()
    models = results.Method.unique()
    table = pd.DataFrame(index=models)

    # Go through the data and create a table with the R2 scores for each methods and variable plus the significance level of the CW test. 
    for var in variables:
        # Subset the data from the current varaible from the dataframe and prepare for concatenation.
        R2 = results[results.Dataset == var]
        R2 = R2.drop(columns=['Dataset', 'CW', 'DA', 'DA HA'])
        R2 = R2.set_index('Method')

        CW = results[results.Dataset == var]
        CW = CW.drop(columns=['Dataset', 'R2', 'DA', 'DA HA'])
        CW = CW.set_index('Method')

        for idx, value in enumerate(CW.CW):
            # If CW value is significant we need to do something
            if('*' in str(value)):
                # Exactract the stars from the value
                stars = re.sub('[0-9]*\.[0-9]*', '', value)

                #Add the stars to the R2 value
                R2.iloc[idx] = str(R2.iloc[idx].values[0]) + str(stars)


        # Rename column to the current varaibles and concatenate to the table
        R2 = R2.rename(columns = {'R2': var})
        table  = pd.concat([table, R2], axis = 1)


    # Rename the model/method names to be in line with those used in the paper. 
    newIndex = replaceNames(results.Method.unique())
    replaceDict = {A: B for A, B in zip(results.Method.unique(), newIndex)}
    table = table.rename(index=replaceDict, inplace = False)
    
    # Rename the variable names to be in line with those used in the paper.
    newColumn = updateVariableNames(results.Dataset.unique(), dataset)
    replaceDict = {A: B for A, B in zip(results.Dataset.unique(), newColumn)}
    table = table.rename(columns=replaceDict, inplace = False)
    return table

table = createResultsTable(resultsMEV, 'MEV')
table

table = createResultsTable(resultsTA, 'TA')
table




