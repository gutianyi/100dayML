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
# @Time    : 26/10/2019 2:10 下午
# @Author  : Q
# @Site    : 
# @File    : lab2_regression.py
# @Software: PyCharm

# from sklearn import linear_model
# x = [[0,0],[1,1],[2,2]]
# y = [0,1,2]
#
# reg = linear_model.LinearRegression()
# reg.fit(x,y)
# print(reg.coef_)
#
# reg1 = linear_model.Ridge(alpha=.5)
# reg1.fit([[0,0], [0,0], [1,1]],[0, .1, 1])
# print(reg1.coef_)
# print(reg1.intercept_)
import copy
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('/Users/elvis/ITProjects/GitHub/PythonTask/100dayML/100_day_ml_py/datasets/Data.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 3].values


#Step 3: Handling the missing data
# from sklearn.preprocessing import Imputer
# imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
# imputer = imputer.fit(X[ : , 1:3])
# X[ : , 1:3] = imputer.transform(X[ : , 1:3])
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])


#Step 4: Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
#Creating a dummy variable
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

X_train_not_change = X_train.copy()
X_test_not_change = X_test.copy()

scaler = StandardScaler()
# Don't cheat - fit only on training data
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# apply same transformation to test data
X_test = scaler.transform(X_test)
print(X_test)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train_1 = sc_X.fit_transform(X_train_not_change)
X_test_1 = sc_X.transform(X_test_not_change)
print("---------------------")
print(X_test_1)