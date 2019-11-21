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
# @Time    : 7/11/2019 2:52 下午
# @Author  : Q
# @Site    : 
# @File    : source_achieve.py
# @Software: PyCharm
import os


def get_file_list(source_dir):
    file_list = []  # 文件路径名列表
    # os.walk()遍历给定目录下的所有子目录，每个walk是三元组(root,dirs,files)
    # root 所指的是当前正在遍历的这个文件夹的本身的地址
    # dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
    # files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
    # 遍历所有评论
    for root, dirs, files in os.walk(source_dir):
        file = [os.path.join(root, filename) for filename in files]
        file_list.extend(file)
    return file_list


def get_label_list(file_list):
    # 提取出标签名
    label_name_list = [file.split("\\")[4] for file in file_list]
    # 标签名对应的数字
    label_list = []
    for label_name in label_name_list:
        if label_name == "neg":
            label_list.append(0)
        elif label_name == "pos":
            label_list.append(1)
    return label_list


'''
获得gensim可用的glove词向量模型
'''
import gensim
import shutil
from sys import platform


# 计算行数，就是单词数
def getFileLineNums(filename):
    f = open(filename, 'r', encoding="utf8")
    count = 0
    for line in f:
        count += 1
    return count


# Linux或者Windows下打开词向量文件，在开始增加一行
def prepend_line(infile, outfile, line):
    with open(infile, 'r', encoding="utf8") as old:
        with open(outfile, 'w', encoding="utf8") as new:
            new.write(str(line) + "\n")
            shutil.copyfileobj(old, new)


def prepend_slow(infile, outfile, line):
    with open(infile, 'r', encoding="utf8") as fin:
        with open(outfile, 'w', encoding="utf8") as fout:
            fout.write(line + "\n")
            for line in fin:
                fout.write(line)


def load(filename):
    num_lines = getFileLineNums(filename)
    gensim_file = '/Users/elvis/ITProjects/GitHub/PythonTask/100dayML/100_day_ml_py/ai_project/glove.model.6B.300d.txt'
    gensim_first_line = "{} {}".format(num_lines, 200)
    # Prepends the line.
    if platform == "linux" or platform == "linux2":
        prepend_line(filename, gensim_file, gensim_first_line)
    else:
        prepend_slow(filename, gensim_file, gensim_first_line)
    model = gensim.models.KeyedVectors.load_word2vec_format(gensim_file)
    return model