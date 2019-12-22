#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-09-26 19:18
# @Author  : Big Q(Quentin)
# @Site    : 
# @File    : test.py
# @Software: PyCharm
"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.4
matplotlib
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


# 这是决策树中的一个典型的数据--鸢尾花
iris = datasets.load_iris()
# 特征数据
iris_feature = iris.data
# 分类数据
iris_target = iris.target
#print iris_target
# 此时数据是按不同的类别顺序排列的，将数据随机的分为训练集和测试集
feature_train, feature_test, target_train, target_test = train_test_split(iris_feature, iris_target, test_size=0.33,
                                                                          random_state=56)
# test_size测试数据占比，一般是70%训练，30%测试
# random_state乱序程度

# 模型训练

# 导入决策树，所有参数为默认，还可以引入损失函数（信息熵，基尼指数）；
# 树深度；叶上最少样本数量，进行裁剪；节点的分裂策略
dt_model = DecisionTreeClassifier()
# 用决策树训练
dt_model.fit(feature_train, target_train)
# 使用测试数据测试
predict_results = dt_model.predict(feature_test)

# 利用测试数据测试
print (predict_results)
print (target_test)

# 以下两种评比测试结果，传入参数有区别
scores = dt_model.score(feature_test, target_test)
print(accuracy_score(predict_results, target_test))