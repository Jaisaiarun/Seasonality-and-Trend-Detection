#%%
import pandas as pd
import datetime
import numpy as np
from statistics import mean
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import ConstantKernel, RBF ,WhiteKernel ,ExpSineSquared , RationalQuadratic
from sklearn.gaussian_process import GaussianProcessRegressor
import seaborn as sns
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')

def Data_Engineering(df):
    df=df.T
    df.columns=['ds','y']
    df.ds = pd.to_datetime(df.ds.apply(str).add('-0'), format='%Y%W-%w')
    plt.figure()
    plt.plot(df.ds,df.y)
    plt.plot(df.ds,df.y, 'r.', markersize=5, label='Observations')
    plt.xlabel('Week')
    plt.xticks(rotation=90)
    plt.ylabel('total transportation volume of the week')
    plt.title('Given Value')
    return df

#%%
def Seasonality_and_Trend_Detection(df,kernel,data_points=6):
    gp = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=9)
    x = np.array(list(range(len(df)))).reshape(-1,1)
    y=df.y.to_numpy().reshape(-1, 1)
    gp.fit(x, y)
    for i in range(data_points):
        t=df["ds"].iloc[-1]+datetime.timedelta(weeks = 1)
        df = df.append({'ds':t,'y':''},ignore_index=True)
    X = np.array(list(range(len(df)))).reshape(-1,1)
    y_pred, sigma = gp.predict(X, return_std=True)
    plt.figure()
#    plt.plot(x, y, 'r:', label='Actual')
#    plt.plot(x, y, 'r.', markersize=10, label='Observations')
    plt.plot(X, y_pred, 'r-', label='Prediction')
    plt.legend(loc='upper left')
    plt.xlabel('Week')
    plt.ylabel('Total transportation volume of the week')
    plt.title('Predicted Value')
    plt.legend(loc='upper left')
    print(y_pred)
    df.y=y_pred
    return df
#%%
k0 = WhiteKernel(noise_level=0.3**2,noise_level_bounds=(0.1**2,0.5**2))
k1 = ConstantKernel(constant_value=2)*ExpSineSquared(length_scale=1.0,periodicity=40)
k2 = ConstantKernel(constant_value=100, constant_value_bounds=(1, 500))*RationalQuadratic(alpha=50.0, length_scale=500)
k3 = ConstantKernel(constant_value=1)*ExpSineSquared(length_scale=1.0,periodicity=12)
kernel_2=k0+k1+k2+k3
k0 = WhiteKernel(noise_level=0.5)
k1 = ConstantKernel(constant_value=2,constant_value_bounds=(0.0, 1.0)) * \
ExpSineSquared(length_scale=1.0, periodicity=40, periodicity_bounds=(35, 45))
k2 = RBF(length_scale=1)  
kernel  = k0 + k1 + k2
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
#%%
df = pd.read_csv('sample.csv', header=None)
df=Data_Engineering(df)
#print(df)

df_1=Seasonality_and_Trend_Detection(df,kernel,10)
#print(df_1)




