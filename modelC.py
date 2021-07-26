import matplotlib.pyplot as plt

import pandas as pd
dfScrn = pd.read_csv('./SPX_OPTIONS_WITH_IVOL_AND_FwdM.csv')
del dfScrn['Unnamed: 0']
dfScrn.head()

dfScrn['tau'] = dfScrn['T'] - dfScrn['t']

dfCall = dfScrn[(dfScrn['Type'] == 'Call') & (dfScrn['FwdM'] >= 0.8) & (dfScrn['FwdM'] <= 1.2)]

fig = plt.figure()

ax = fig.add_subplot(projection='3d')
ax.scatter(dfCall['tau'], dfCall['FwdM'], dfCall['Ivol']*100, marker='.', c='k')
ax.set_xlabel('Tau')
ax.set_ylabel('Forward Moneyness')
ax.set_zlabel('BS Implied Volatility(%)')
plt.title("Implied Volatility Scatter Call(Call)")

plt.show()

x = dfCall['tau'].copy()
y = dfCall['FwdM'].copy()
z = dfCall['Ivol'].copy()

import numpy as np
from scipy import interpolate
xnew = np.linspace(x.min(), x.max(), len(x) * 3, endpoint=True)
ynew = np.linspace(y.min(), y.max(), len(y) * 3, endpoint=True)
xnew_edges, ynew_edges = np.meshgrid(xnew, ynew)

tck = interpolate.bisplrep(x, y, z, s=100)

znew = interpolate.bisplev(xnew, ynew, tck)

spline = interpolate.Rbf(x,y,z, function='thin_plate')

znew = spline(xnew_edges, ynew_edges)

fig = plt.figure()

ax = plt.axes(projection='3d')
ax.plot_wireframe(xnew_edges, ynew_edges, znew*100)
ax.plot_surface(xnew_edges, ynew_edges, znew*100, alpha=0.2)
ax.scatter3D(x,y,z*100, c='r')

ax.set_xlabel('Tau')
ax.set_ylabel('Forward Moneyness')
ax.set_zlabel('BS Implied Volatility(%)')
plt.title("Implied Volatility Surface with (Call)")

plt.show()

fig = plt.figure()

ax = plt.axes(projection='3d')
ax.plot_wireframe(xnew_edges, ynew_edges, znew*100)
ax.plot_surface(xnew_edges, ynew_edges, znew*100, alpha=0.2)
ax.scatter3D(x,y,z*100, c='r')

ax.set_xlabel('Tau')
ax.set_ylabel('Forward Moneyness')
ax.set_zlabel('BS Implied Volatility(%)')
plt.title("Implied Volatility Surface with (Call)")

plt.show()
