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
# @Time    : 7/11/2019 11:17 下午
# @Author  : GU Tianyi
# @File    : pytorch_cnn_mnist.py
import pickle

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
# data loading and transforming
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable

data_transform = transforms.ToTensor()

# prepare data
train_data = MNIST(root='./data', train=True, transform=data_transform, download=False)
test_data = MNIST(root='./data', train=False, transform=data_transform, download=False)

print('train_data number: ', len(train_data))
print('Test_data number: ', len(test_data))

# batch size
batch_size = 40
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


# make a cnn
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, 3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
        )
        self.fc1 = torch.nn.Linear(20 * 5 * 5, 50)
        self.fc1_drop = torch.nn.Dropout(p=0.4)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        in_fc = out_conv2.view(out_conv2.size(0), -1)
        out_fc1 = self.fc1(in_fc)
        out_drop = self.fc1_drop(out_fc1)
        out = self.fc2(out_fc1)
        return out


my_cnn = Net()

# print(my_cnn)

# 定义优化器
optimizer = torch.optim.SGD(my_cnn.parameters(), lr=0.01, momentum=0.9)
loss_fun = torch.nn.CrossEntropyLoss()


# train

def train_fun(EPOCH):
    loss_list = []
    for eopch in range(EPOCH):
        for step, data in enumerate(train_loader):
            b_x, b_y = data
            b_x, b_y = Variable(b_x), Variable(b_y)
            out_put = my_cnn(b_x)
            loss = loss_fun(out_put, b_y)
            if step % 100 == 0:
                print('Epoch: ', eopch, ' Step: ', step, ' loss: ', float(loss))
                loss_list.append(float(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return loss_list


def save():
    torch.save(my_cnn, '/Users/elvis/ITProjects/GitHub/PythonTask/100dayML/100_day_ml_py/ai_project/pyTorch_data_store/first_train.pkl')


def restore():
    torch.load('/Users/elvis/ITProjects/GitHub/PythonTask/100dayML/100_day_ml_py/ai_project/pyTorch_data_store/first_train.pkl')


# test

def test_fun():
    correct = 0
    test_loss = torch.zeros(1)
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    for batch_i, data in enumerate(test_loader):
        # get the input images and their corresponding labels
        inputs, labels = data
        print(inputs)
        print(labels)
        # wrap them in a torch Variable
        # volatile means we do not have to track how the inputs change
        inputs, labels = Variable(inputs, volatile=True), Variable(labels, volatile=True)
        # forward pass to get outputs
        outputs = my_cnn(inputs)
        # calculate the loss
        loss = loss_fun(outputs, labels)
        # update average test loss
        test_loss = test_loss + ((torch.ones(1) / (batch_i + 1)) * (loss.data - test_loss))
        # get the predicted class from the maximum value in the output-list of class scores
        _, predicted = torch.max(outputs.data, 1)
        # compare predictions to true label
        # print(predicted)
        # print(labels)
        correct += np.squeeze(predicted.eq(labels.data.view_as(predicted))).sum()
        acc = correct / (40 * batch_i + 1)
        print(' Step: ', batch_i + 1, ' loss: ', float(loss), 'accurary: ', acc)


# 预测可以用下面这个来计算
# # Calculate accuracy before training
# correct = 0
# total = 0
#
# # Iterate through test dataset
# for images, labels in test_loader:
#
#     # forward pass to get outputs
#     # the outputs are a series of class scores
#     outputs = net(images)
#
#     # get the predicted class from the maximum value in the output-list of class scores
#     _, predicted = torch.max(outputs.data, 1)
#
#     # count up total number of correct labels
#     # for which the predicted and true labels are equal
#     total += labels.size(0)
#     correct += (predicted == labels).sum()

# # calculate the accuracy
# accuracy = 100 * correct / total

# print it out!
# print('Accuracy before training: ', accuracy)

if __name__ == '__main__':
    # training_loss=train_fun(3)
    # print(training_loss)
    # plt.plot(training_loss)
    # plt.xlabel('1000\'s of batches')
    # plt.ylabel('loss')
    # plt.ylim(0, 2.5) # consistent scale
    # plt.show()
    # save()
    # restore()
    test_fun()