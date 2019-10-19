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
# @File    : linear_regression.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

dataset = pd.read_csv('./raw_price_train/1_r_price_train.csv',index_col='Date')
dataset.index = pd.to_datetime(dataset.index)
test_set = dataset['2015-12-10':'2015-12-20']
dataset = dataset['2012-01-01':'2015-12-10']
dataset['Adj Close'].diff().plot(grid = True)
plt.show()

# 白噪声监测
from statsmodels.stats.diagnostic import acorr_ljungbox
print('白噪声检测结果是 ', acorr_ljungbox(dataset['Adj Close'], lags= 1))

# from statsmodels.tsa.arima_model import ARIMA
# pmax = 3
# qmax = 3
# dataset['Adj Close'] = dataset['Adj Close'].astype(float)

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

train_model = ARIMA(dataset,(p,1,q)).fit()
pre = 20
yHat = train_model.forecast(pre)
print(yHat[0])
print(test_set)
