#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-09-30 22:12
# @Author  : Big Q(Quentin)
# @Site    : 
# @File    : GreetingServer.py
# @Software: PyCharm

import socket
import threading
import time


def tcplink(sock, addr):
    print('Accept new connection from %s:%s...' % addr)
    sock.send(b'Connect successfully! \nWelcome, Client! plz give me your name!')
    while True:
        data = sock.recv(1024)
        print(str(data))
        if not data or data.decode('utf-8') == 'exit':
            break
        sock.send(('Hello, %s!' % data.decode('utf-8')).encode('utf-8'))
    sock.close()
    print('Connection from %s:%s closed.' % addr)


serverName, port = "127.0.0.1", 9891

mSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
mSocket.bind((serverName, port))
mSocket.listen(1)
print("Waiting for client on port "+ str(port) + "...")
while True:
    # 接受一个新连接:
    sock, addr = mSocket.accept()
    # 创建新线程来处理TCP连接:
    t = threading.Thread(target=tcplink, args=(sock, addr))
    t.start()


