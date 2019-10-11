#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-09-30 22:12
# @Author  : Big Q(Quentin)
# @Site    :
# @File    : GreetingClient.py
# @Software: PyCharm

import socket
from sys import stdin

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 建立连接:
s.connect(('127.0.0.1', 9891))
print('Start to connect to local server on 9891 by TCP')
# 接收server端数据
print(s.recv(1024).decode('utf-8'))
value = stdin.readline().strip('\n')
# print(value)
s.send(str(value).encode())
print(s.recv(1024).decode('utf-8'))
to_cipher_text = stdin.readline().strip('\n')
s.send(str(to_cipher_text).encode())
print(s.recv(1024).decode('utf-8'))
s.send(b'exit')
s.close()

