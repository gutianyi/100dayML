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
# @Time    : 17/11/2019 5:06 下午
# @Author  : GU Tianyi
# @File    : pytorch_cnn_imdb_self.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
import torch.autograd as autograd
import torchtext.vocab as torchvocab
from torch.autograd import Variable
import tqdm
import os
import time
import re
import pandas as pd
import string
import gensim
import time
import random
import snowballstemmer
import collections
from collections import Counter
from nltk.corpus import stopwords
from itertools import chain
from sklearn.metrics import accuracy_score

def readIMDB(path, seg='train'):
    pos_or_neg = ['pos', 'neg']
    data = []
    for label in pos_or_neg:
        files = os.listdir(os.path.join(path, seg, label))
        for file in files:
            with open(os.path.join(path, seg, label, file), 'r', encoding='utf8') as rf:
                review = rf.read().replace('\n', '')
                if label == 'pos':
                    data.append([review, 1])
                elif label == 'neg':
                    data.append([review, 0])
    return data

train_data = readIMDB('/Users/elvis/ITProjects/GitHub/PythonTask/100dayML/100_day_ml_py/ai_project/data/imdb/aclImdb')
test_data = readIMDB('/Users/elvis/ITProjects/GitHub/PythonTask/100dayML/100_day_ml_py/ai_project/data/imdb/aclImdb', 'test')

def tokenizer(text):
    return [tok.lower() for tok in text.split(' ')]


# 接着是分词，这里只做非常简单的分词，也就是按照空格分词。当然按照一些传统的清洗方式效果会更好。
train_tokenized = []
test_tokenized = []
for review, score in train_data:
    train_tokenized.append(tokenizer(review))
for review, score in test_data:
    test_tokenized.append(tokenizer(review))

vocab = set(chain(*train_tokenized))
vocab_size = len(vocab)


# 因为这个数据集非常小，所以如果我们用这个数据集做word embedding有可能过拟合，而且模型没有通用性，所以我们传入一个已经学好的word embedding。
wvmodelwvmodel = gensim.models.KeyedVectors.load_word2vec_format('test_word.txt',
                                                          binary=False, encoding='utf-8')



