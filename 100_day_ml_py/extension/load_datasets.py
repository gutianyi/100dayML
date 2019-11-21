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
# @Time    : 8/11/2019 3:04 下午
# @Author  : GU Tianyi
# @File    : load_datasets.py

from torchtext import data
from torchtext.data import Dataset
from torchtext import datasets
from torchtext.vocab import GloVe

from nltk import word_tokenize
import time
import dill
from config import *


class SNLIDataset(Dataset):
    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(
            len(ex.premise), len(ex.hypothesis))


class SNLI(object):
    def __init__(self, args):
        start_time = time.clock()
        # 定义如何处理数据和标签
        print('1定义如何处理文本和标签')
        self.TEXT = data.Field(batch_first=True,
                               include_lengths=True,  # 是否（返回）一个包含最小batch的句子长度
                               tokenize=word_tokenize,  # 分词
                               lower=True)  # 数据转换成小写

        self.LABEL = data.Field(sequential=False,  # 是否把数据表示成序列，如果是False, 不能使用分词
                                unk_token=None)  # unk的默认为<unk>,todo 为什么改变
        step1_time = time.clock()
        print('\t耗时%.4fs' % (step1_time - start_time))

        # 划分数据集
        print('2划分数据集')

        if self.if_split_already():
            print('从本地加载划分好的数据集...')
            fields = {'premise': self.TEXT, 'hypothesis': self.TEXT, 'label': self.LABEL}
            self.train, self.dev, self.test = self.load_split_datasets(fields)
        else:
            print('本地没有发现数据集,开始划分数据集...')
            self.train, self.dev, self.test = datasets.SNLI.splits(self.TEXT, self.LABEL, root='.data')
            self.dump_examples(self.train, self.dev, self.test)

        step2_time = time.clock()
        print('\t耗时%.4fs' % (step2_time - step1_time))

        # 创建词汇表
        print('3创建词汇表')
        # self.TEXT.build_vocab(self.train, self.dev, self.test, vectors=GloVe(name='840B', dim=300))
        # self.LABEL.build_vocab(self.train)
        if os.path.exists(snli_text_vocab_path) and os.path.exists(snli_label_vocab_path):
            print('加载已创建的词汇表...')
            with open(snli_text_vocab_path, 'rb')as f:
                self.TEXT.vocab = dill.load(f)
            with open(snli_label_vocab_path, 'rb')as f:
                self.LABEL.vocab = dill.load(f)
        else:
            print('本地没有发现词汇表,新建词汇表...')
            self.TEXT.build_vocab(self.train, self.dev, self.test, vectors=GloVe(name='840B', dim=300))
            self.LABEL.build_vocab(self.train)
            with open(snli_text_vocab_path, 'wb')as f:
                dill.dump(self.TEXT.vocab, f)
            with open(snli_label_vocab_path, 'wb')as f:
                dill.dump(self.LABEL.vocab, f)

        step3_time = time.clock()
        print('\t耗时%.4fs' % (step3_time - step2_time))

        # 生成Batch迭代器
        print('4生成Batch迭代器')
        self.train_iter, self.dev_iter, self.test_iter = \
            data.BucketIterator.splits((self.train, self.dev, self.test),
                                       batch_size=args.batch_size,
                                       device=args.gpu)
        step4_time = time.clock()
        print('\t耗时%.4fs' % (step4_time - step3_time))

    # 判断是否已经划分好数据并保存到本地
    # 全部3个文件全部存在则返回True；否则返回False
    def if_split_already(self):
        for path in snli_split_path_lst:
            if not os.path.exists(path):
                return False
        return True

    # 从本地加载切分好的数据集
    def load_split_datasets(self, fields):
        # 加载examples
        with open('/Users/elvis/ITProjects/GitHub/PythonTask/100dayML/100_day_ml_py/ai_project/pyTorch_data_store/snli_train_examples_path', 'rb')as f:
            train_examples = dill.load(f)
        with open('/Users/elvis/ITProjects/GitHub/PythonTask/100dayML/100_day_ml_py/ai_project/pyTorch_data_store/snli_dev_examples_path', 'rb')as f:
            dev_examples = dill.load(f)
        with open('/Users/elvis/ITProjects/GitHub/PythonTask/100dayML/100_day_ml_py/ai_project/pyTorch_data_store/snli_test_examples_path', 'rb')as f:
            test_examples = dill.load(f)

        # 恢复数据集
        train = SNLIDataset(examples=train_examples, fields=fields)
        dev = SNLIDataset(examples=dev_examples, fields=fields)
        test = SNLIDataset(examples=test_examples, fields=fields)
        return train, dev, test

    # 将切分好的数据集保存到本地
    def dump_examples(self, train, dev, test):
        # 保存examples
        if not os.path.exists('/Users/elvis/ITProjects/GitHub/PythonTask/100dayML/100_day_ml_py/ai_project/pyTorch_data_store/snli_train_examples_path'):
            with open('/Users/elvis/ITProjects/GitHub/PythonTask/100dayML/100_day_ml_py/ai_project/pyTorch_data_store/snli_train_examples_path', 'wb')as f:
                dill.dump(train.examples, f)
        if not os.path.exists('/Users/elvis/ITProjects/GitHub/PythonTask/100dayML/100_day_ml_py/ai_project/pyTorch_data_store/snli_dev_examples_path'):
            with open('/Users/elvis/ITProjects/GitHub/PythonTask/100dayML/100_day_ml_py/ai_project/pyTorch_data_store/snli_dev_examples_path', 'wb')as f:
                dill.dump(dev.examples, f)
        if not os.path.exists('/Users/elvis/ITProjects/GitHub/PythonTask/100dayML/100_day_ml_py/ai_project/pyTorch_data_store/snli_test_examples_path'):
            with open('/Users/elvis/ITProjects/GitHub/PythonTask/100dayML/100_day_ml_py/ai_project/pyTorch_data_store/snli_test_examples_path', 'wb')as f:
                dill.dump(test.examples, f)