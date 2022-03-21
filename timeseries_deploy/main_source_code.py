## time series as a regression problem using ridge regression

##import libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from statsmodels.tsa.stattools import adfuller

#read data
data=pd.read_csv('gold_rate_10January22.csv')
print('dataset head....')
print(data.head())


#drop 24k columns and rename the columns as goldprice
data.drop(columns=['Pure Gold (24 k)'],inplace=True)
data=data.rename(columns={"Standard Gold (22 K)":"goldprice"})

#consider latest 124 (3 months) rows for forecasting
#format the date column as pandas date time
#Set Date column index
#drop Date column
data=data.iloc[-124:,:]
data['Date']=pd.to_datetime(data['Date'],format='%Y-%m-%d')
data.set_index(data['Date'],inplace=True)
data.drop(columns=['Date'],inplace=True)

#print the dataframe shape
print(f"Number observations in dataframe={data.shape[0]}")

#plot the series
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(data['goldprice'],label='gold price')
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.set_title('Latest 124 days Gold Price')
ax.set_xlabel("Months")
ax.set_ylabel("Gold Price")
ax.legend()
ax.grid()
plt.show()

#histogram and density plots for timeseries
data['goldprice'].plot.density()
plt.title('Density plot for goldprice')
plt.show()

data['goldprice'].hist()
plt.title('Histogram for goldprice')
plt.show()


#boxplot-for months september to december we check how median and mean is varying
groups=data['2021-09-01':'2021-12-31']['goldprice']
groups.index.names = ['Months']
sns.boxplot(y=groups,x=groups.index.month_name(),showmeans=True,meanprops={"markerfacecolor": "white","markeredgecolor": "white"})
plt.title('Boxplt for Months')
plt.show()


#checks for time dependency structure using summary statistics
split=int(len(data)*0.5)
print("summary statistics....")
mean1,mean2=data[:split].values.mean(),data[split:].values.mean()
var1,var2=data[:split].values.var(),data[split:].values.var()
print(f"mean 1={mean1}----mean 1={mean2}")
print(f"variance 1={var1}----variance 2={var2}")

#augmented dicky-fuller test in order to check the time series is stationary/non stationary

result=adfuller(data['goldprice'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
  print('\t%s: %.3f' % (key, value))
if result[1]>0.05:
    print("p_value > 0.05, it has unit root and non-stationary,Fail to reject the null hypothesis")
else:
    print("p_value <= 0.05, it has no unit root and stationary,Reject the null hypothesis")

#function to create lagged dataset/array using sliding window method
def create_lagged_df(no_lags:int,base_df):
    df=base_df.copy()
    j=0
    for i in range(no_lags,0,-1):
        df.insert(loc=j,column='t-'+str(i),value=base_df.shift(i))
        j+=1
    df.dropna(inplace=True)
    return df.values

#function to split the data as train and test for chain walk validation
def splitdata_train_test(data,no_test_obs):
    train,test=data[:-no_test_obs],data[-no_test_obs:]
    return train,test

#function for ridge regession forecast
def ridge_regression_forecast(train,testX,alpha):
    train=np.array(train)
    X_train,y_train=train[:,:-1],train[:,-1]
    model=Ridge(alpha=alpha,solver='auto',random_state=1)
    model.fit(X_train,y_train)
    predict=model.predict([testX])
    return predict[0]

#function for walk forward method, by considering  one step at a time
def walkforward_validation(data,no_lags,no_tests_obs,alpha):
    data_array=create_lagged_df(no_lags,data)
    train,test=splitdata_train_test(data_array,no_tests_obs)
    predictions=list()
    history=[x for x in train]
    for i in range(len(test)):
        X_test,y_test=test[i,:-1],test[i,-1]
        predict=ridge_regression_forecast(history,X_test,alpha)
        predictions.append(predict)
        history.append(test[i])
    rmse=np.sqrt(mean_squared_error(test[:,-1],predictions))
    return rmse,predictions,test[:,-1]

#gridsearch for ridgeregression hyper-parameters in order obtain best rmse
test_samples=[int(len(data)*x) for x in [0.1,0.15,0.20]]
lags=list(range(1,21))
alphas=[0.0001,0.001,0.01,.05,0.10,0.25,0.5,0.75,1.0,1.25,1.5,2.0]
rmse_score=list()
rmse_config=list()
for lag in lags:
    for no_test in test_samples:
        for alpha in alphas:
            rmse,_,_=walkforward_validation(data,lag,no_tests_obs=no_test,alpha=alpha)
            rmse_score.append(rmse)
            rmse_config.append((lag,no_test,alpha))
            print(f"config(lag,no_test,alpha)=({lag},{no_test},{alpha})---->rmse={rmse:.3f}")

#print the min rmse and corresponding hyperparameter
min_rmse=min(rmse_score)
best_config=rmse_config[rmse_score.index(min_rmse)]
print(f"best_rmse score is {min_rmse:.3f} and corresponding configuration is {best_config[0]} lags,{best_config[1]} no_test and {best_config[2]} alpha...")

#run configuration which resulted in lesser rmse
rmse,predictions,y_test=walkforward_validation(data,no_lags=best_config[0],no_tests_obs=best_config[1],alpha=best_config[2])
print(f"rmse {rmse:3f}")
print('\n')
plt.plot(y_test,label='expected')
plt.plot(predictions,label='predictions')
plt.title("Expected vs Predicted")
plt.ylabel("goldprice")
plt.xlabel("test observations")
plt.legend()
plt.grid()
plt.show()


#plot residuals (check stationary/non-stationary,check temporal structure in residuals)
residual=[i-j for i,j in zip(y_test,predictions)]
plt.plot(residual,label='residuals')
plt.title('Residuals')
plt.grid(True)
plt.legend()
plt.show()

print(f" residual mean ={np.mean(np.array(residual))}")

#Final model
data_final=create_lagged_df(5,data)
X,y=data_final[:,:-1],data_final[:,-1]
final_model=Ridge(alpha=0.1,solver='auto',random_state=1)
final_model.fit(X,y)
