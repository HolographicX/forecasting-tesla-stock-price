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

fig, ax = plt.subplots()
ax.set_xlabel('time')
ax.set_ylabel('amplitude')
plt.title("Tesla stock prices 2016-2020")

time = np.arange(1, 1000, 1)

# tesla stock prices from 
file = open('/run/media/soham/4024229A242292C8/Main/python/tcs internship/time series/tesla stock forecast/tesla-stock-2016-2020.csv', 'rb')
y = loadtxt(file,delimiter = ",")
print(y)

# plot_pacf(y, lags=39, title=" Autocorrelation - Non-Stationary process 1")
# plt.scatter(time, y)
# plt.show()

'''
-------------
--- ARIMA ---
------------- 
'''
y_arima = ARIMA(y, order=(1,0,0)).fit()

x_start = 500;

x_end = 1000;

y_hat = y_arima.predict(x_start,x_end)

plt.plot(y, linewidth = 1, label = "Actual")

plt.plot(list(range(x_start, x_end+1)), y_hat, color= "blue",linestyle="dotted", linewidth = 2, label = "Predicted with (1, 0, 0)")

plt.legend()

plt.show()
# train = y[0:800]
# test = y[-199:]
# # plt.plot(train)
# plt.plot(test)
# plt.plot(train)
# plt.show()
# model = ARIMA(train, order = (1, 1, 1))
  
# result = model.fit()
# print(result.summary())

# timeplot = time[0:800]
# yplot = y[0:800]
# plt.plot(timeplot, yplot)
# plt.plot(result.predict(10,80), color='red')

# y0 = y[1:999]
# y1 = y[0:998]
# z = []
# for i in range(len(y0)):
#     z.append(y0[i] - y1[i])
    


# plt.plot(z)
# plt.show()