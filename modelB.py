import numpy as np
from scipy.stats import norm

def BS_call(t, S, K, T, r, sigma):
    tau = T-t
    # print(np.sqrt(tau))
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * norm.cdf(d1) - np.exp(-r * tau) * K * norm.cdf(d2)

def BS_put(t, S, K, T, r, sigma):
    tau = T-t
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*tau) / (sigma*np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return np.exp(-r * tau) * K * norm.cdf(-d2) - S * norm.cdf(-d1)

def BS_vega(t, S, K, T, r, sigma):
    tau = T - t
    # print(sigma * np.sqrt(tau))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    return S * norm.pdf(d1) * np.sqrt(tau)

def NewtonRaphson_sigma_call(target_value, t, S, K, T, r, *args):
    max_iter = 1000
    precision = 1.0e-8
    sigma = 0.1
    # print(target_value, t, S, K, T, r)
    for i in range(max_iter):
        price = BS_call(t, S, K, T, r, sigma)
        vega = BS_vega(t, S, K, T, r, sigma)
        diff = target_value - price
        if (abs(diff) < precision):
            return sigma
        # print(vega)
        sigma = sigma + diff/vega
    return sigma

def NewtonRaphson_sigma_put(target_value, t, S, K, T, r, *args):
    max_iter = 1000
    precision = 1.0e-8
    sigma = 0.2
    for i in range(max_iter):
        price = BS_put(t, S, K, T, r, sigma)
        vega = BS_vega(t, S, K, T, r, sigma)
        diff = target_value - price
        if (abs(diff) < precision):
            return sigma
        sigma = sigma + diff/vega
    return sigma

def Forward_Moneyness(target_value, t, S, K, T, r, Type, sigma):
    F = Forward_Price(t, S, T, r)
    tau = T - t
    return np.log(F/K) / (sigma*np.sqrt(tau))

def Forward_Price(t, S, T, r):
    return S*np.exp(r*(T-t))

def NewtonRaphson_sigma(target_value, t, S, K, T, r, Type):
  if Type == 'Call':
    return NewtonRaphson_sigma_call(target_value, t, S, K, T, r)
  return NewtonRaphson_sigma_put(target_value, t, S, K, T, r)

from google.colab import files
files.upload()

import pandas as pd

df0621 = pd.read_csv('./$spx-options-exp-2021-06-21-weekly-show-all-stacked-05-22-2021.csv', thousands=',')
df0621 = df0621[:-1]
df0621['t'] = 141/365
df0621['T'] = 172/365
df0621['r'] = (1+0.06)/(1+0.02) - 1
df0621['St'] = 4155.86
df0621['Strike'] = df0621['Strike'].astype(str).str.replace(',', '').astype(float)
dfExt0621 = df0621[['Midpoint', 't', 'St', 'Strike', 'T', 'r', 'Type']]
dfExt0621.head()

dfExt0621.tail()

files.upload()

df0730 = pd.read_csv('$spx-options-exp-2021-07-30-weekly-show-all-stacked-05-22-2021.csv', thousands=',')
df0730 = df0730[:-1]
df0730['t'] = 141/365
df0730['T'] = 211/365
df0730['r'] = (1+0.06)/(1+0.02) - 1
df0730['St'] = 4155.86
df0730['Strike'] = df0730['Strike'].astype(str).str.replace(',', '').astype(float)
dfExt0730 = df0730[['Midpoint', 't', 'St', 'Strike', 'T', 'r', 'Type']]
dfExt0730.head()

files.upload()

df0917 = pd.read_csv('$spx-options-exp-2021-09-17-weekly-show-all-stacked-05-22-2021.csv', thousands=',')
df0917 = df0917[:-1]
df0917['t'] = 141/365
df0917['T'] = 260/365
df0917['r'] = (1+0.06)/(1+0.02) - 1
df0917['St'] = 4155.86
df0917['Strike'] = df0917['Strike'].astype(str).str.replace(',', '').astype(float)
dfExt0917 = df0917[['Midpoint', 't', 'St', 'Strike', 'T', 'r', 'Type']]
dfExt0917.head()

dfStacked = pd.concat([dfExt0621, dfExt0730, dfExt0917], ignore_index=True)
dfStacked.head()

dfStacked.tail()

dfStacked['Ivol'] = dfStacked.apply(lambda x: NewtonRaphson_sigma(*x), axis=1)

dfScrn = dfStacked[dfStacked['Ivol'].notna()]
dfScrn.shape

dfScrn.head()

dfScrn['FwdM'] = dfScrn.apply(lambda x: Forward_Moneyness(*x), axis=1)

dfScrn['FwdM'].min(), dfScrn['FwdM'].max()

dfScrn.to_csv('SPX_OPTIONS_WITH_IVOL_AND_FwdM.csv')

files.download('SPX_OPTIONS_WITH_IVOL_AND_FwdM.csv')

from google.colab import files

files.upload()

import pandas as pd
dfScrn = pd.read_csv('SPX_OPTIONS_WITH_IVOL_AND_FwdM.csv')
del dfScrn['Unnamed: 0']
dfScrn.head()

import matplotlib.pyplot as plt

dfScrn['tau'] = dfScrn['T'] - dfScrn['t']

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(dfScrn['tau'], dfScrn['FwdM'], dfScrn['Ivol'], marker='o')

dfCall = dfScrn[(dfScrn['Type'] == 'Call') & (dfScrn['FwdM'] >= 0.8) & (dfScrn['FwdM'] <= 1.2)]


fig = plt.figure()

ax = fig.add_subplot(projection='3d')
ax.scatter(dfCall['tau'], dfCall['FwdM'], dfCall['Ivol']*100, marker='.', c='k')
ax.set_xlabel('Tau')
ax.set_ylabel('Forward Moneyness')
ax.set_zlabel('BS Implied Volatility(%)')
plt.title("Implied Volatility Scatter Call(Put)")

plt.show()

dfPut = dfScrn[(dfScrn['Type'] == 'Call') & (dfScrn['FwdM'] >= 0.8) & (dfScrn['FwdM'] <= 1.2)]

fig = plt.figure()

ax = fig.add_subplot(projection='3d')
ax.scatter(dfPut['tau'], dfPut['FwdM'], dfPut['Ivol']*100, marker='.', c='k')
ax.set_xlabel('Tau')
ax.set_ylabel('Forward Moneyness')
ax.set_zlabel('BS Implied Volatility(%)')
plt.title("Implied Volatility Scatter Plot(Put)")

plt.show()
