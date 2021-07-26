import pandas as pd
df = pd.read_csv('./$spx-options-exp-2021-06-21-weekly-near-the-money-stacked-05-22-2021.csv', thousands=',')

df.head()

df['t'] = 141/365
df['T'] = 172/365

df['St'] = 4155.86
df['r'] = 1.06/1.02 - 1

import numpy as np
from scipy.stats import norm

def BS_call(t, S, K, T, r, sigma):
    tau = T-t
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
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    return S * norm.pdf(d1) * np.sqrt(tau)

def NewtonRaphson_sigma_call(target_value, t, S, K, T, r, *args):
    max_iter = 1000
    precision = 1.0e-8
    sigma = 0.1
    for i in range(max_iter):
        price = BS_call(t, S, K, T, r, sigma)
        vega = BS_vega(t, S, K, T, r, sigma)
        diff = target_value - price
        if (abs(diff) < precision):
            return sigma
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

dfCall = df[['Midpoint', 't', 'St', 'Strike', 'T', 'r']]
dfCall = dfCall[:20]
dfCall['Strike'] = dfCall['Strike'].str.replace(',', '').astype(float)
dfCall.tail()

dfCall['Ivol'] = dfCall.apply(lambda x: NewtonRaphson_sigma_call(*x), axis=1)

dfCall.head()

dfCall.tail()

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca()
ax.set_xticks(np.arange(4050, 4250, 50))
ax.set_yticks(np.arange(11.5, 16.5, 1))
plt.plot(dfExt['Strike'], dfExt['Ivol']*100, 'o')
plt.xlabel('K'); plt.ylabel('BS Implied Vol(%)')
plt.grid()
plt.axis([4050, 4250, 11.5, 16.5])
plt.show()

dfPut = df[['Midpoint', 't', 'St', 'Strike', 'T', 'r']]
dfPut = dfPut[-21:-1]
dfPut['Strike'] = dfPut['Strike'].str.replace(',', '').astype(float)
dfPut.tail()

dfPut['Ivol'] = dfPut.apply(lambda x: NewtonRaphson_sigma_put(*x), axis=1)

dfPut

fig = plt.figure()
ax = fig.gca()
ax.set_xticks(np.arange(4050, 4250, 50))
ax.set_yticks(np.arange(13, 20, 2))
plt.plot(dfPut['Strike'], dfPut['Ivol']*100, 'o')
plt.xlabel('K'); plt.ylabel('BS Implied Vol(%)')
plt.grid()
plt.axis([4050, 4250, 13, 20])
plt.show()

files.upload()

df = pd.read_csv('$spx-options-exp-2021-06-21-weekly-show-all-stacked-05-22-2021.csv', thousands=',')
df = df[:-1]
df['t'] = 141/365
df['T'] = 172/365
df['r'] = (1+0.06)/(1+0.02) - 1
df['St'] = 4155.86

dfCall = df[df.Type == 'Call']
dfCall = dfCall[['Midpoint', 't', 'St', 'Strike', 'T', 'r']]
# dfCall = dfCall[6:90]
dfCall['Strike'] = dfCall['Strike'].astype(str).str.replace(',', '').astype(float)

NewtonRaphson_sigma_call(909.75, 0.386301, 4155.86, 3250.0, 0.471233, 1.039216)

dfCall.head()

dfCall.dtypes

dfCall.head()

NewtonRaphson_sigma_call(132.6, 0.386301, 4155.86, 4075.0, 0.471233, 1.039216)

dfCall.head()

dfCall['Ivol'] = dfCall.apply(lambda x: NewtonRaphson_sigma_call(*x), axis=1)

dfCall = dfCall[dfCall['Ivol'].notna()]

dfCall['Strike'].min(), dfCall['Strike'].max()

dfCall['Ivol'].min(), dfCall['Ivol'].max()

dfCall = dfCall[(dfCall['Strike'] >= 4000) & (dfCall['Strike'] <= 4500)]

dfCall.head()


fig = plt.figure()
ax = fig.gca()
ax.set_xticks(np.arange(3950, 4600, 100))
ax.set_yticks(np.arange(10, 19, 1))
plt.plot(dfCall['Strike'], dfCall['Ivol']*100, 'o')
plt.xlabel('K'); plt.ylabel('BS Implied Vol(%)')
plt.grid()
plt.axis([3950, 4600, 10, 19])
plt.show()

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_reg = PolynomialFeatures(degree=4)
X = dfCall['Strike'].values.reshape(-1, 1)
y = (dfCall['Ivol'] * 100).values.reshape(-1, 1)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

# Visualizing the Polymonial Regression results
def viz_polymonial():
    plt.scatter(X, y, color='red')
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
    plt.title('Implied Volatility Curve(Call)')
    plt.xlabel('Strike')
    plt.ylabel('BS Implied Vol(%)')
    plt.show()
    return
viz_polymonial()

dfCall['est.'] = BS_call(dfCall['t'], dfCall['St'], dfCall['Strike'], dfCall['T'], dfCall['r'], pol_reg.predict(X_poly).reshape(-1)/100)

plt.plot(dfCall['Strike'], dfCall['Midpoint'], 'ko', label='Observed Call Price')
plt.plot(dfCall['Strike'], dfCall['est.'], 'r-', label='Estimated Curve')
plt.legend(loc='upper right', numpoints=1)
plt.show()
plt.savefig('Option Price Curve.png')

dfK = pd.DataFrame(X_poly)
dfK.head()

dfPut = df[df.Type == 'Call']
dfPut = dfPut[['Midpoint', 't', 'St', 'Strike', 'T', 'r']]
dfPut['Strike'] = dfPut['Strike'].astype(str).str.replace(',', '').astype(float)

dfPut.head()

dfPut['Ivol'] = dfPut.apply(lambda x: NewtonRaphson_sigma_put(*x), axis=1)

dfPut = dfPut[dfPut['Ivol'].notna()]

dfPut.head()

dfPut['Strike'].min(), dfPut['Strike'].max()

dfPut['Ivol'].min(), dfPut['Ivol'].max()

dfPut = dfPut[(dfPut['Strike'] >= 4000) & (dfPut['Strike'] <= 4500)]

fig = plt.figure()
ax = fig.gca()
ax.set_xticks(np.arange(3950, 4250, 100))
ax.set_yticks(np.arange(2, 60, 10))
plt.plot(dfPut['Strike'], dfPut['Ivol']*100, 'o')
plt.xlabel('K'); plt.ylabel('BS Implied Vol(%)')
plt.grid()
plt.axis([3950, 4250, 2, 60])
plt.show()

from sklearn.linear_model import LinearRegression
X = dfPut['Strike'].values.reshape(-1, 1)
y = (dfPut['Ivol']*100).values.reshape(-1, 1)
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Visualizing the Linear Regression results
def viz_linear():
    plt.scatter(X, y, color='red')
    plt.plot(X, lin_reg.predict(X), color='blue')
    plt.title('Implied Volatility Curve(Put)')
    plt.xlabel('Strike')
    plt.ylabel('BS Implied Vol(%)')
    plt.show()
    return
viz_linear()

dfPut['est.'] = BS_put(dfPut['t'], dfPut['St'], dfPut['Strike'], dfPut['T'], dfPut['r'], lin_reg.predict(X).reshape(-1)/100)

plt.plot(dfPut['Strike'], dfPut['Midpoint'], 'ko', label='Observed Call Price')
plt.plot(dfPut['Strike'], dfPut['est.'], 'r-', label='Estimated Curve')
plt.legend(loc='upper right', numpoints=1)
plt.show()
plt.savefig('Put Option Price Curve.png')

