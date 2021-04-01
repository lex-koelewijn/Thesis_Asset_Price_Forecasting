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
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn

ep = pd.read_excel('data/Augemented_Formatted_results.xls', sheet_name='Equity premium', skiprows= range(1118,1127,1))[:-1]
ep['Date'] = pd.to_datetime(ep['Date'], format='%Y%m')
ep = ep.set_index('Date')

ep

rawData =  pd.read_excel('data/Returns_econ_tech_data_augmented.xls', sheet_name='Monthly')

spot = rawData[['Date', 'S&P 500 index']]
spot['Date'] = pd.to_datetime(spot['Date'], format='%Y%m')
spot = spot.set_index('Date')

spot

#Select data starting from 1950-12 to be in line with Rapach
spot = spot.loc[(spot.index >= '1950-12-01')]

plt.figure(figsize=(24,12))
plt.plot(spot, label = 'S&P 500 Spot Index Level', linewidth = .75)
plt.legend()
plt.title('S&P 500 Spot Index Level', fontweight = 'bold')

plt.figure(figsize=(24,12))
# plt.plot(ep['Simple equity premium'], label = 'S&P 500 Simple equity premium', linewidth = .75)
# plt.plot(ep['Log equity premium'], label = 'S&P 500 Log equity premium', linewidth = .75)
# plt.plot((spot['S&P 500 index']/spot['S&P 500 index'].shift(1))-1, label = 'S&P500 Monthly return')
plt.plot(np.log(spot['S&P 500 index'])-np.log(spot['S&P 500 index'].shift(1)), label = 'S&P500 Monthly return')
# plt.plot(spot['S&P 500 index'].diff(1))
plt.legend()
plt.title('S&P 500 Monthly Returns', fontweight = 'bold')

# +
mpl.rcParams["font.size"] = 14
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24,12))

ax[0].plot(spot, label = 'S&P 500 Spot Index Level', linewidth = .75)
ax[0].legend()
ax[0].title.set_text('S&P 500 Index Level')

ax[1].plot(np.log(spot['S&P 500 index'])-np.log(spot['S&P 500 index'].shift(1)), label = 'S&P500 Monthly return')
ax[1].legend()
ax[1].title.set_text('S&P 500 Monthly Returns')

plt.savefig("plots/SP500.png", bbox_inches='tight', dpi=500)
# -




