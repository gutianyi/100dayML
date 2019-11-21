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
# -*- coding:" utf-8" -*-
# @Time    : 19/10/2019 8:10 下午
# @Author  : Q
# @Site    :
# @File    : try_PyTorch.py
# @Software: PyCharm
# import torch
# x = torch.ones(2, 2, requires_grad=True)
# print(x)
# y = x + 1
# z = y*y*2
# out = z.mean()
# print(z,out)
# out.backward()
# print(x.grad )


import pandas as pd
import numpy as np
import json
import datetime
import os

path = "./stock_dataset_v3/tweet_train/1_tweet_train"
files_name = os.listdir(path)
key_words_list = []
for file_name in files_name:
    txt = pd.read_csv('./stock_dataset_v3/tweet_train/1_tweet_train/%s' % file_name, header=None, delimiter="\t")
    for item in txt.values:
        key_words_list.append(
            [datetime.datetime.strptime(json.loads(item[0])['created_at'], "%a %b %d %H:%M:%S +0000 %Y").date(),
             json.loads(item[0])['text']])

key_words_df = pd.DataFrame(key_words_list).sort_values(by=0)
key_words_df.reset_index(drop=True, inplace=True)

print(key_words_df)








# txt = pd.read_csv('./stock_dataset_v3/tweet_train/1_tweet_train/2014-01-01',header=None , delimiter="\t")
# key_words_list = []
# for item in txt.values:
#     key_words_list.append([datetime.datetime.strptime(json.loads(item[0])['created_at'] ,"%a %b %d %H:%M:%S +0000 %Y").date(),
#                            json.loads(item[0])['text']])
#
# key_words_df = pd.DataFrame(key_words_list)
# print(key_words_df)