#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 13/10/2019 7:06 下午
# @Author  : Q
# @Site    : 
# @File    : stock_arima.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.arima_model import ARIMA

dataset = pd.read_csv('./raw_price_train/1_r_price_train.csv',index_col='Date')
dataset.index = pd.to_datetime(dataset.index)
# test_set = dataset['2015-12-12':'2015-12-20']
dataset = dataset['2012-01-01':'2015-12-20']
dataset['Adj Close'].diff().plot(grid = True)
plt.show()

dataset['Adj Close'].plot(grid = True)
plt.title('Stock1\'s Adj Close',fontsize='large')
plt.savefig('Adj Close.jpg')
plt.show()

dataset['Volume'].plot(grid = True)
plt.title('Stock1\'s Volume',fontsize='large')
plt.savefig('Volume.jpg')
plt.show()

# 白噪声监测
from statsmodels.stats.diagnostic import acorr_ljungbox
# print('白噪声检测结果是 ', acorr_ljungbox(dataset['Adj Close'], lags= 1))

from statsmodels.tsa.stattools import adfuller as ADF
for diff in range(1,50):
    diff_data = dataset['Adj Close'].diff(periods=diff).dropna()
    x = ADF(diff_data)[1]
    y = acorr_ljungbox(diff_data, lags= 1)[1][0]
    print(diff, '阶差分ADF', x)
    print(diff, '阶差分白噪声', y)
    if (x < 0.05 and y < 0.05):
        break
'''
1 阶差分ADF 0.0
1 阶差分白噪声 0.9470182792171196
2 阶差分ADF 1.3453557791437636e-07
2 阶差分白噪声 1.6014367935309312e-42
'''

pmax = 3
qmax = 3
dataset['Adj Close'] = dataset['Adj Close'].astype(float)

bic_matrix = []
for p in range(pmax + 1):
    tmp = []
    for q in range(qmax + 1):
        try:
            tmp.append(ARIMA(dataset['Adj Close'], (p,2,q)).fit().bic)
        except:
            tmp.append(None)
        bic_matrix.append(tmp)

#找出最小值
bic_matrix = pd.DataFrame(bic_matrix)
print(bic_matrix)
p,q = bic_matrix.stack().idxmin()
print(u'BIC 最小的p值 和 q 值：%s,%s' %(p,q))
train_model = ARIMA(dataset['Adj Close'],(p,2,q)).fit()
# 预测未来几天Adj Close
yHat = train_model.forecast(7)[0]
print('yHat', yHat)
# print('test_set: ',test_set)

#
# def loop_train(dataset,i):
#     loop_train_model = ARIMA(dataset['Adj Close'], (0, 1, 1)).fit()
#     dataset['Adj Close'].loc[datetime.datetime(2015,12,12+i)] = loop_train_model.forecast(1)[0][0]
#     return loop_train_model.forecast(1)[0]
# i = 0
# while i < 7 :
#     loop_train(dataset,i)
#     i = i + 1

from sklearn.metrics import mean_squared_error
print(mean_squared_error(test_set['Adj Close'], yHat))