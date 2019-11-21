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
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/11/2019 1:41 下午
# @Author  : Q
# @Site    : 
# @File    : ml_project_svr_predict_7_days.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.svm import SVR

dataset = np.array(pd.read_csv('./raw_price_train/1_r_price_train.csv'))
dataset = np.delete(dataset, [1,2,3,4], axis=1)
# x_five_item_train = []
# ## 前七天预测后七天(有两天是双休)
# while len(dataset) > 5:
#     x_five_item = []
#     for _ in range(5):
#         x_five_item.append(dataset[0])
#         dataset = np.delete(dataset,0,axis=0)
#     x_five_item_train.append(x_five_item)
#
# y_five_item_train = np.array(x_five_item_train[1:])
# x_five_item_train = np.array(x_five_item_train[:-1])
# svr = SVR(kernel='poly')
#
# y_pred = svr.fit(x_five_item_train,y_five_item_train)

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
dataset[:,0] = labelencoder_X.fit_transform(dataset[:,0])
y_train = dataset[:, -2]
x_train = np.delete(dataset,-2,axis=1)

svr = SVR(kernel='poly')
svr.fit(x_train,y_train)
svr.predict([''])

