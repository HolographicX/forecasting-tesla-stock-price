import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from numpy import loadtxt
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.graphics.tsaplots import plot_acf

def difference(dataset, interval=1):
 diff = list()
 for i in range(interval, len(dataset)):
    value = dataset[i] - dataset[i - interval]
    diff.append(value)
 return diff

fig, ax = plt.subplots()
ax.set_xlabel('time (days from 2016-02-12)')
ax.set_ylabel('closing price')
plt.title("Tesla stock prices 2016-2020")

time = np.arange(1, 1000, 1)

# tesla stock prices from 
file = open('/run/media/soham/4024229A242292C8/Main/python/tcs internship/time series/tesla stock forecast/tesla-stock-2016-2020.csv', 'rb')
y = loadtxt(file,delimiter = ",")
# print(y)

# plot_pacf(y, lags=39, title=" Autocorrelation - Non-Stationary process 1")
# plt.scatter(time, y)
# plt.show()


'''
-------------
--- ARIMA ---
------------- 
'''
'''
---
'''
plot_acf(y, lags=999, title="Autocorrelation Function (ACF) - Tesla Stock Prices (2016-2020)")
plot_acf(difference(y), lags=998, title="Autocorrelation Function (ACF) - diff(Tesla Stock Prices (2016-2020))")

plt.show()
data = y

train_size = int(len(data) * 0.66)
train, test = data[0:train_size], data[train_size:len(data)]
history = [x for x in train]
predictions = list()
print(data[train_size-10:train_size+3])

for t in range(len(test)):    
    model = ARIMA(history, order=(2,1,2))
    model_fit = model.fit()
    
    pred = model_fit.forecast()
    yhat = pred[0]
    predictions.append(yhat)    # Append test observation into overall record
    obs = test[t]
    history.append(obs)
    if t < 6: 
        print(model_fit.summary())
        print(t, " prediction vs actual ", list(data).index(obs) , str(pred), str (obs) )

plt.plot(data, color= "grey", linewidth = 2, label = "Actual")
plt.plot( list(range(train_size, len(data))), predictions, linestyle="dotted", linewidth = 3, label = "Predicted with (2,1,2)")
#list(range(train_size, len(data))),
plt.legend()

plt.show()
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
print("max: " + str(max(y)))
print("min: " + str(min(y)))




