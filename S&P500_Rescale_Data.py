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

# # Data scaling
# This notebook is used to prepare the S&P500 volume data to the correct format. We can obtain daily numbers but we need the volume of the last day of each month. The script below creates a new excel with this output dynamically.

import pandas as pd
import numpy as np

data = pd.read_csv('data/GSPC_Daily_2011-2019.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
data = data.set_index('Date')

# +
# data
# -

#Groupby year and month and take the last value of each month of each year. 
dataDownSampled = data.groupby([data.index.year, data.index.month]).tail(1)

# +
# dataDownSampled
# -

dataDownSampled.to_excel('data/S&P500_End_Of_Month.xlsx')


