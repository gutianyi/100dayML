#                       _oo0oo_
#                      o8888888o
#                      88" . "88
#                      (| -_- |)
#                      0\  =  /0
#                    ___/`---'\___
#                  .' \\|     |// '.
#                 / \\|||  :  |||// \
#                / _||||| -:- |||||- \
#               |   | \\\  -  /// |   |
#               | \_|  ''\---/''  |_/ |
#               \  .-\__  '-'  ___/-. /
#             ___'. .'  /--.--\  `. .'___
#          ."" '<  `.___\_<|>_/___.' >' "".
#         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#         \  \ `_.   \_ __\ /__ _/   .-` /  /
#     =====`-.____`.___ \_____/___.-`___.-'=====
#                       `=---='
#
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#               佛祖保佑         永无BUG
#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 13/10/2019 7:06 下午
# @Author  : Q
# @Site    : 
# @File    : ml_project_arima_predict_7_days.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.arima_model import ARIMA

dataset = pd.read_csv('./raw_price_train/1_r_price_train.csv',index_col='Date')
dataset.index = pd.to_datetime(dataset.index)
test_set = dataset['2015-12-12':'2015-12-20']
dataset = dataset['2012-01-01':'2015-12-11']
dataset['Adj Close'].diff().plot(grid = True)
plt.show()

# 白噪声监测
from statsmodels.stats.diagnostic import acorr_ljungbox
print('白噪声检测结果是 ', acorr_ljungbox(dataset['Adj Close'], lags= 1))

pmax = 3
qmax = 3
dataset['Adj Close'] = dataset['Adj Close'].astype(float)

# bic_matrix = []
# for p in range(pmax + 1):
#     tmp = []
#     for q in range(qmax + 1):
#         try:
#             tmp.append(ARIMA(dataset['Adj Close'], (p,2,q)).fit().bic)
#         except:
#             tmp.append(None)
#         bic_matrix.append(tmp)
#
# #找出最小值
# bic_matrix = pd.DataFrame(bic_matrix)
# print(bic_matrix)
# p,q = bic_matrix.stack().idxmin()
p,q = 0, 1
print(u'BIC 最小的p值 和 q 值：%s,%s' %(p,q))

# train_model = ARIMA(dataset['Adj Close'],(p,2,q)).fit()
#预测未来几天Adj Close
# yHat = train_model.forecast(1)[0]
# print(yHat)
# print(test_set)
# dataset['Adj Close'].loc[datetime.datetime(2015,12,11)] = 999
# print(dataset['Adj Close'])


def loop_train(dataset,i):
    loop_train_model = ARIMA(dataset['Adj Close'], (0, 1, 1)).fit()
    dataset['Adj Close'].loc[datetime.datetime(2015,12,12+i)] = loop_train_model.forecast(1)[0][0]
    return loop_train_model.forecast(1)[0]
i = 0
while i < 7 :
    loop_train(dataset,i)
    i = i + 1


print(dataset['Adj Close'].tail(5))
from sklearn.metrics import mean_squared_error
print(mean_squared_error(test_set['Adj Close'], dataset['Adj Close'].tail(5)))