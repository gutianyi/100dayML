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
# @Time    : 20/11/2019 12:39 下午
# @Author  : GU Tianyi
# @File    : pytorch_lstm_imdb.py
from itertools import chain

import numpy as np

np.random.seed(10)
import re

re_tag = re.compile(r'<[^>]+>')


def rm_tags(text):
    return re_tag.sub('', text)

import os


def read_files(filetype):
    path = "/Users/elvis/ITProjects/GitHub/PythonTask/100dayML/100_day_ml_py/ai_project/data/imdb/aclImdb/"
    file_list = []

    positive_path = path + filetype + "/pos/"
    for f in os.listdir(positive_path):
        file_list += [positive_path + f]

    negative_path = path + filetype + "/neg/"
    for f in os.listdir(negative_path):
        file_list += [negative_path + f]

    print('read', filetype, 'files:', len(file_list))

    all_labels = ([1] * 12500 + [0] * 12500)

    all_texts = []

    for fi in file_list:
        with open(fi, encoding='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]

    return all_labels, all_texts


y_train, train_text = read_files("train")
y_test, test_text = read_files("test")

# 取词
def tokenizer(text):
    return [tok.lower() for tok in text.split(' ')]

train_tokenized = []
test_tokenized = []
for review, score in train_text:
    train_tokenized.append(tokenizer(review))
for review, score in test_text:
    test_tokenized.append(tokenizer(review))

vocab = set(chain(*train_tokenized))
vocab_size = len(vocab)

# 先读取所有文章建立字典，限制字典的数量为nb_words=2000
#
#
# token = Tokenizer(num_words=3800)
# token.fit_on_texts(train_text)
#
# # 将文字转为数字序列
#
#
# x_train_seq = token.texts_to_sequences(train_text)
# x_test_seq = token.texts_to_sequences(test_text)
#
# # 截长补短，让所有影评所产生的数字序列长度一样
#
# x_train = sequence.pad_sequences(x_train_seq, maxlen=380)
# x_test = sequence.pad_sequences(x_test_seq, maxlen=380)