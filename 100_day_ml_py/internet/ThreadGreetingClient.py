#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-09-30 22:12
# @Author  : Big Q(Quentin)
# @Site    : 
# @File    : GreetingClient.py
# @Software: PyCharm

import socket
from sys import stdin

# serverName = "127.0.0.1"
# port = 9891
#
# print("The client is connecting to " + serverName + " on port " + str(port))
#
# mSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# mSocket.connect((serverName, port))

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 建立连接:
s.connect(('127.0.0.1', 9891))
print('Start to connect to local server on 9891 by TCP')
# 接收欢迎消息:
print(s.recv(1024).decode('utf-8'))
value = stdin.readline().strip('\n')
# print(value)
s.send(str(value).encode())
print(s.recv(1024).decode('utf-8'))
s.send(b'exit')
s.close()

# mSocket.close()
