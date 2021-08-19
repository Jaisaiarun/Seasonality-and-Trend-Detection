#%%
import pandas as pd
import datetime
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
#%%
def Data_Engineering(df):
    df=df.T
    df.columns=['ds','y']
    df.ds = pd.to_datetime(df.ds.apply(str).add('-0'), format='%Y%W-%w')
    return df
#%%
def f(x):    
        return x * np.sin(x)
#%%
def Seasonality_and_Trend_Detection(df,data_points=6):
    
    return df
#%%
df = pd.read_csv('sample.csv', header=None)
#df=df.T
#df.columns=['ds','y']
#df.ds = pd.to_datetime(df.ds.apply(str).add('-0'), format='%Y%W-%w')
df=Data_Engineering(df)
print(df)
#%%
#print(np.array(lis(df)))
#%%
#plt.plot(list(range(len(df))),df.y)
#dates = matplotlib.dates.date2num(df.ds)
plt.plot(df.ds,df.y)
plt.xticks(rotation=90) #turn x label 90
plt.show()
#print(df.ds)
#%%
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
#X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
#x = np.atleast_2d(np.linspace(0, 10, 1000)).T
x = np.array(list(range(len(df)))).reshape(-1,1)
#x = df.ds.to_numpy().reshape(-1,1)
#X=x

# Observations
#y = f(X).ravel()
y=df.y.to_numpy().reshape(-1, 1)
gp.fit(X, y)
# Make the prediction on the meshed x-axis (ask for MSE as well)
#y_pred, sigma = gp.predict(df.y.to_numpy().reshape(-1, 1), return_std=True)
y_pred, sigma = gp.predict(x, return_std=True)

#%%
# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure()
#plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
plt.plot(X, y, 'r.', markersize=10, label='Observations')
plt.plot(x, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
        np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None' )
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
#plt.ylim(-10, 20)
plt.legend(loc='upper left')

