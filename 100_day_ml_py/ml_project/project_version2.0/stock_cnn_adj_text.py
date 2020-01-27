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
# @Time    : 22/12/2019 1:23 下午
# @Author  : GU Tianyi
# @File    : stock_cnn_adj_vol.py

import pandas as pd
import torch
import pickle
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler ##数据归一化

output_size = 7


def preprocess(data, m, n):
    '''
    data: the dataframe of stock price
    m: 前m天预测
    n: 预测几天
    '''
    minMax = MinMaxScaler()
    data_transformed = minMax.fit_transform(data)

    adj_close = data["Adj Close"].tolist()
    adj_volume = data_transformed[:, -2:]
    res_X = []
    res_y = []

    for i in range(0, len(adj_close) - m - n + 1):
        res_X.append(adj_volume[i:i + m])
        res_y.append(adj_close[i + m: i + m + n])
    return res_X, res_y, adj_volume[-m:]


dataset = pd.read_csv(
    "raw_price_train/1_r_price_train.csv",
    index_col='Date')
dataset.index = pd.to_datetime(dataset.index)


X, y, final_pred = preprocess(dataset, 14, output_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
print('X size',(np.array(X)).shape)
# print(y[-1])


import torch.utils.data as Data

tensor_x_train = torch.Tensor(X_train)
tensor_y_train = torch.Tensor(y_train)

tensor_x_test = torch.Tensor(X_test)
tensor_y_test = torch.Tensor(y_test)

# 先转换成 torch 能识别的 Dataset
torch_train_dataset = Data.TensorDataset(tensor_x_train, tensor_y_train)
torch_test_dataset = Data.TensorDataset(tensor_x_test, tensor_y_test)

# 把 dataset 放入 DataLoader
train_loader = Data.DataLoader(
    dataset=torch_train_dataset,  # torch TensorDataset format
    batch_size=100,  # mini batch size
    shuffle=False,  # 要不要打乱数据 (打乱比较好)
    num_workers=2,  # 多线程来读数据
)
test_loader = Data.DataLoader(
    dataset=torch_test_dataset,  # torch TensorDataset format
    batch_size=100,  # mini batch size
    shuffle=False,  # 要不要打乱数据 (打乱比较好)
    num_workers=2,  # 多线程来读数据
)


import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape ( None, 14, 2)
            nn.Conv1d(
                in_channels=14,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,  # 如果想要 con2d 出来的图片长宽没有变化,当 stride=1 padding=(kernel_size-1)/2
            ),
            nn.ReLU(),  # activation
            nn.MaxPool1d(kernel_size=2),  # 在 2x2 空间里向下采样, output shape (batch_size, 16, 1)
        )
        self.out = nn.Sequential(
            #             nn.Linear(16, 4),
            #             nn.Dropout(dropout),
            #             nn.ReLU(),
            nn.Linear(16, output_size),
        )

    def forward(self, x):
        conv1ed = self.conv1(x)  # 展平多维的卷积图成 (batch_size, 16, 1)
        #         print('conv1ed size ',conv1ed.size())
        conv1ed = conv1ed.view(conv1ed.size(0), -1)
        #         print('conv1ed viewed size',conv1ed.size())
        output = self.out(conv1ed)
        return output.squeeze()

model = CNN()
print(model)


import torch.optim as optim

# 定义优化器
optimizer = optim.Adam(model.parameters())
loss_func = nn.MSELoss()

from sklearn.metrics import mean_squared_error


def train(model, loader, optimizer, loss_func):
    model.train()

    for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
        # 训练的地方...
        batch_x = batch_x.view(-1, 14, 2)  # reshape x to (batch, time_step, input_size)
        batch_y = batch_y.view(-1, output_size)
        output = model(batch_x)
        loss = loss_func(output, batch_y)  # cross entropy loss

        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()

        acc = mean_squared_error(batch_y.numpy().tolist(), output.detach().numpy())

    return loss.data.numpy(), acc

# 不用优化器了
def evaluate(model, loader, loss_func):
    # 转成测试模式，冻结dropout层或其他层
    model.eval()

    with torch.no_grad():
        for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习

            batch_x = batch_x.view(-1, 14, 2)  # reshape x to (batch, time_step, input_size)
            batch_y = batch_y.view(-1, output_size)
            output = model(batch_x)
            loss = loss_func(output, batch_y)
            acc = mean_squared_error(batch_y.numpy().tolist(), output.detach().numpy())

        # 调回训练模式
        model.train()

    return acc


best_test_mse = float('inf')
for epoch in range(3):  # 训练批次数
    train_loss, train_mse = train(model, train_loader, optimizer, loss_func)
    test_mse = evaluate(model, test_loader, loss_func)

    if test_mse < best_test_mse:
        best_test_mse = test_mse
        torch.save(model.state_dict(), 'stock-cnn-model.pt')

    if epoch > 0:
        print('Epoch: ', epoch, '| train_loss: ', train_loss, '| train_mse x: ', train_mse, '| test_mse x: ', test_mse)
        # 打出来一些数据
#         print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
#               batch_x.numpy(), '| batch y: ', batch_y.numpy())


model.load_state_dict(torch.load('stock-cnn-model.pt'))
pred = model(torch.Tensor(final_pred).view(1, 14, 2)).detach().numpy()
print(pred)

for i in range(len(pred)):
    file_name = 'pkl_files/stock1_day%d.pkl' % (i+1)
    # pickle a variable to a file
    file = open(file_name, 'wb')
    pickle.dump(pred[i], file)
    file.close()

print("输出pkl完成")
