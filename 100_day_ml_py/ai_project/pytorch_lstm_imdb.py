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
# @File    : pytorch_bidirectional_lstm_imdb.py

import torch
import torch.nn.functional as F
from torchtext import data
from torchtext import datasets
import time
import random
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    total_len = 0

    # model.train()代表了训练模式
    # model.train() ：启用 BatchNormalization 和 Dropout
    # model.eval() ：不启用 BatchNormalization 和 Dropout
    model.train()

    # iterator为train_iterator
    for batch in iterator:
        # 梯度清零，加这步防止梯度叠加
        optimizer.zero_grad()

        # batch.text 就是上面forward函数的参数text
        # 压缩维度，不然跟 batch.label 维度对不上
        predictions = model(batch.text)

        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)

        loss.backward()  # 反向传播
        optimizer.step()  # 梯度下降

        # loss.item() 以及本身除以了 len(batch.label)
        # 所以得再乘一次，得到一个batch的损失，累加得到所有样本损失
        epoch_loss += loss.item() * len(batch.label)

        # (acc.item(): 一个batch的正确率) * batch数 = 正确数
        # train_iterator 所有batch的正确数累加
        epoch_acc += acc.item() * len(batch.label)

        # 计算 train_iterator 所有样本的数量，应该是17500
        total_len += len(batch.label)
        # print('train loss = ', epoch_loss / total_len, '| train acc = ', epoch_acc / total_len)

    # epoch_loss / total_len ：train_iterator所有batch的损失
    # epoch_acc / total_len ：train_iterator所有batch的正确率
    return epoch_loss / total_len, epoch_acc / total_len
# 计算预测的准确率

def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


# 不用优化器了
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    total_len = 0

    # 转成测试模式，冻结dropout层或其他层
    model.eval()

    with torch.no_grad():
        # iterator为valid_iterator
        for batch in iterator:
            # 没有反向传播和梯度下降

            predictions = model(batch.text)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item() * len(batch.label)
            epoch_acc += acc.item() * len(batch.label)
            total_len += len(batch.label)

    # 调回训练模式
    model.train()

    return epoch_loss / total_len, epoch_acc / total_len

import time

# 查看每个epoch的时间
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs




#================================================分隔符=================================================================


import torch
from torchtext import data
import torch.nn as nn
import torch.nn.functional as F

SEED = 1234
BATCH_SIZE = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED)  # 为CPU设置随机种子
torch.cuda.manual_seed(SEED)  #为GPU设置随机种子
# 在程序刚开始加这条语句可以提升一点训练速度，没什么额外开销
torch.backends.cudnn.deterministic = True

# 首先，我们要创建两个Field 对象：这两个对象包含了我们打算如何预处理文本数据的信息。
# spaCy:英语分词器,类似于NLTK库，如果没有传递tokenize参数，则默认只是在空格上拆分字符串。
# torchtext.data.Field : 用来定义字段的处理方法（文本字段，标签字段）
TEXT = data.Field(tokenize='spacy',fix_length=380)
#LabelField是Field类的一个特殊子集，专门用于处理标签。
LABEL = data.LabelField(dtype=torch.float)

# 加载IMDB电影评论数据集
from torchtext import datasets
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

import random
# 默认split_ratio=0.7
train_data, valid_data = train_data.split(random_state=random.seed(SEED))
print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

# 从预训练的词向量（vectors）中，将当前(corpus语料库)词汇表的词向量抽取出来，构成当前 corpus 的 Vocab（词汇表）
# 预训练的 vectors 来自glove模型，每个单词有100维。glove模型训练的词向量参数来自很大的语料库
# 而我们的电影评论的语料库小一点，所以词向量需要更新，glove的词向量适合用做初始化参数。
TEXT.build_vocab(train_data, max_size=3800, vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
LABEL.build_vocab(train_data)

print(f'Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}')
print(f'Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}')



# 相当于把样本划分batch，知识多做了一步，把相等长度的单词尽可能的划分到一个batch，不够长的就用padding。
train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    device = device
)

import torch.nn as nn


'''
LSTM(
  (embedding): Embedding(5002, 50)
  (lstm): LSTM(50, 64, num_layers=2, dropout=0.2)
  (fc): Linear(in_features=64, out_features=2, bias=True)
)
'''


class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(len(TEXT.vocab), embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=2,
                            dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        packed_output, (hidden, cell) = self.lstm(embedded)
        return self.fc(packed_output[-1,:,:]).view(-1)

pretrained_embeddings = TEXT.vocab.vectors
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5


PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]  # PAD_IDX = 1 为pad的索引

model = RNN( EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)

# model.embedding.weight.data.copy_(pretrained_embeddings)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]   # UNK_IDX = 0

# 词汇表25002个单词，前两个unk和pad也需要初始化，把它们初始化为0
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

import torch.optim as optim

# 定义优化器
optimizer = optim.Adam(model.parameters())

# 定义损失函数，这个BCEWithLogitsLoss特殊情况，二分类损失函数
criterion = nn.BCEWithLogitsLoss()

# 送到GPU上去
model = model.to(device)
criterion = criterion.to(device)

N_EPOCHS = 10
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'lstm-model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

