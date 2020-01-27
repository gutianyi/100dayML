#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 22/12/2019 1:23 下午
# @Author  : GU Tianyi
# @File    : stock_cnn_adj_vol.py

import pandas as pd
import torch
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  ##数据归一化

output_size = 14
m = 7
n = output_size



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
    #
    res_X = []
    res_y = []

    for i in range(0, len(adj_close) - m - n + 1):
        res_X.append(adj_volume[i:i + m])
        res_y.append(adj_close[i + m:i + m + n])
    return res_X, res_y, adj_volume[-m:]


dataset = pd.read_csv('raw_price_train/8_r_price_train.csv', index_col='Date')
dataset.index = pd.to_datetime(dataset.index)



X, y, final_pred = preprocess(dataset, m, output_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
# print((np.array(X_train)).shape)
# print((np.array(y_train)).shape)
# print((np.array(X)).shape)
# print((np.array(y)).shape)
# print((np.array(y[-1])).shape)
# print(np.array(final_pred).shape)



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
    shuffle=True,  # 要不要打乱数据 (打乱比较好)
    num_workers=2,  # 多线程来读数据
)
test_loader = Data.DataLoader(
    dataset=torch_test_dataset,  # torch TensorDataset format
    batch_size=100,  # mini batch size
    shuffle=True,  # 要不要打乱数据 (打乱比较好)
    num_workers=2,  # 多线程来读数据
)

import torch.nn as nn


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(  # if use nn.RNN(), it hardly learns
            input_size=2,
            hidden_size=64,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(64, output_size)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)  # None represents zero initial hidden state

        # choose r_out at the last time step
        #         print('r_out[:, -1, :]',r_out[:, -1, :].size())
        out = self.out(r_out[:, -1, :])
        return out


class BILSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=2,
            hidden_size=64,  # rnn hidden unit
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.2)
        self.fc = nn.Sequential(
            #             nn.Linear(64*2, 32),
            #             nn.Dropout(0.2),
            #             nn.ReLU(),
            nn.Linear(64 * 2, output_size),
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.bilstm(x, None)  # None represents zero initial hidden state
        #         print('r_out',r_out.size())
        #         print('h_n',h_n.size())
        #         print('h_n[-2, :, :]',h_n[-2, :, :].size())
        #         print('h_n[-2, :, :],h_n[-1, :, :]),dim=1:',torch.cat((h_n[-2, :, :],h_n[-1, :, :]),dim=1).size())
        output = self.fc(torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1))
        return output




model = BILSTM()
import torch.optim as optim

# 定义优化器
optimizer = optim.Adam(model.parameters())
loss_func = nn.MSELoss()



from sklearn.metrics import mean_squared_error


def train(model, loader, optimizer, loss_func):
    model.train()

    for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
        # 训练的地方...
        batch_x = batch_x.view(-1, m, 2)  # reshape x to (batch, time_step, input_size)
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

            batch_x = batch_x.view(-1, m, 2)  # reshape x to (batch, time_step, input_size)
            batch_y = batch_y.view(-1, output_size)
            output = model(batch_x)
            loss = loss_func(output, batch_y)
            acc = mean_squared_error(batch_y.numpy().tolist(), output.detach().numpy())

        # 调回训练模式
        model.train()

    return acc


best_test_mse = float('inf')
for epoch in range(1501):  # 训练所有!整套!数据 10 次
    train_loss, train_mse = train(model, train_loader, optimizer, loss_func)
    test_mse = evaluate(model, test_loader, loss_func)

    if test_mse < best_test_mse:
        best_test_mse = test_mse
        torch.save(model.state_dict(), 'stock-bilstm-model.pt')

    if epoch % 100 == 0:
        print('Epoch: ', epoch, '| train_loss: ', train_loss, '| train_mse x: ', train_mse, '| test_mse x: ', test_mse)


model.load_state_dict(torch.load('stock-bilstm-model.pt'))
pred = model(torch.Tensor(X).view(-1, m, 2)).detach().numpy()
print(pred)


pred_list = np.array([])
print(pred_list)
for i in range(1, len(pred)):
    pred_list = np.append(pred_list, pred[i][-1])

print(pred_list.shape)
plt.title('BiLSTM-Adj Close_Vol-Result')
plt.plot(pred_list, color='green', label='predicted Adj Close')
plt.plot(dataset['Adj Close'].tolist(), color='red', label='Adj Close')

from collections import OrderedDict
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())# 显示图例

plt.savefig('BiLSTM-Adj Close_Vol-Result.jpg')
plt.show()

print(model(torch.Tensor(final_pred).view(1, m, 2)).detach().numpy())


pred = model(torch.Tensor(final_pred).view(1, m, 2))
print(model)
print(pred)