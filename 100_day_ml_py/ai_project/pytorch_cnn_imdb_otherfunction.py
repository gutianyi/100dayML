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
# @Time    : 8/11/2019 2:37 下午
# @Author  : GU Tianyi
# @File    : pytorch_cnn_imdb_otherfunction.py
import torch
import random
import time
import spacy
from _pytest import logging
from torch.utils.data import DataLoader
from torchtext import data
from torchtext import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# 1）下载数据，train & test
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize='spacy', fix_length=38)
LABEL = data.LabelField(dtype=torch.float)

print("\ndowning IMDB data...")
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
print(type(train_data))
print('finished...')

print('---------------')
print(vars(train_data.examples[0]))
print('---------------')

# 2) 切分数据 train valid
print("\n切分数据 train valid data...")
train_data, valid_data = train_data.split(random_state=random.seed(SEED))
print("Number of training examples: ", len(train_data))
print("Number of validation examples: ", len(valid_data))
print("Number of testing examples: ", len(test_data))
print('finished...')

# 3) build vocab， 并定义max_size  & glove
print("\nbuild vocab， 并定义max_size  & glove...")
TEXT.build_vocab(train_data, max_size=3800)
LABEL.build_vocab(train_data)

print('---------------')
print("TEXT.vocab.freqs.most_common(20)", TEXT.vocab.freqs.most_common(20))
print('---------------')
print("TEXT.vocab.itos[:10]", TEXT.vocab.itos[:10])
print('---------------')
print('finished')

# 4) 创建 iterator batch-examples
print("\n创建 iterator batch-examples...")
BATCH_SIZE = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device
)
print('finished')


# cnn model
class TextCNN(nn.Module):
    def __init__(self, vec_dim, filter_num, sentence_max_size, label_size, kernel_list):
        """
        :param vec_dim: 词向量的维度
        :param filter_num: 每种卷积核的个数
        :param sentence_max_size:一篇文章的包含的最大的词数量
        :param label_size:标签个数，全连接层输出的神经元数量=标签个数
        :param kernel_list:卷积核列表
        """
        super(TextCNN, self).__init__()
        chanel_num = 1
        # nn.ModuleList相当于一个卷积的列表，相当于一个list
        # nn.Conv1d()是一维卷积。in_channels：词向量的维度， out_channels：输出通道数
        # nn.MaxPool1d()是最大池化，此处对每一个向量取最大值，所有kernel_size为卷积操作之后的向量维度
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(chanel_num, filter_num, (kernel, vec_dim)),
            nn.ReLU(),
            # 经过卷积之后，得到一个维度为sentence_max_size - kernel + 1的一维向量
            nn.MaxPool2d((sentence_max_size - kernel + 1, 1))
        )
            for kernel in kernel_list])
        # 全连接层，因为有2个标签
        self.fc = nn.Linear(filter_num * len(kernel_list), label_size)
        # dropout操作，防止过拟合
        self.dropout = nn.Dropout(0.5)
        # 分类
        self.sm = nn.Softmax(0)

    def forward(self, x):
        # Conv2d的输入是个四维的tensor，每一位分别代表batch_size、channel、length、width
        in_size = x.size(0)  # x.size(0)，表示的是输入x的batch_size
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(in_size, -1)  # 设经过max pooling之后，有output_num个数，将out变成(batch_size,output_num)，-1表示自适应
        out = F.dropout(out)
        out = self.fc(out)  # nn.Linear接收的参数类型是二维的tensor(batch_size,output_num),一批有多少数据，就有多少行
        print('out是', out)
        return out

def train_textcnn_model(net, iterator, epoch):
    print("begin training")
    net.train()  # 必备，将模型设置为训练模式
    # optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer = optim.Adam(net.parameters())
    criterion = nn.CrossEntropyLoss()
    for i in range(epoch):  # 多批次循环
        for batch in iterator: # 有多少个batch
            optimizer.zero_grad()  # 清除所有优化的梯度
            output = net(batch.text)  # 传入数据并前向传播获取输出
            print("batch.text.size()",batch.text.size())
            loss = criterion(output, batch.label)
            loss.backward()
            optimizer.step()
            # 打印状态信息
            logging.info("train epoch=" + str(i) + ",loss=" + str(loss.item() / 64))
    print('Finished Training')


sentence_max_size = 300  # 每篇文章的最大词数量
batch_size = 64
filter_num = 100  # 每种卷积核的个数
kernel_list = [3, 4, 5]  # 卷积核的大小
label_size = 2
vec_dim = 300
input = torch.randn(batch_size, 1, sentence_max_size, 300)
net = TextCNN(vec_dim, filter_num, sentence_max_size, label_size, kernel_list)
output = net(input)

train_textcnn_model(net, train_iterator, 5)
