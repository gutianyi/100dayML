#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-09-30 22:12
# @Author  : Big Q(Quentin)
# @Site    : 
# @File    : GreetingServer.py
# @Software: PyCharm

import socket
import base64


def encodeobj(obj_bytes):
    return base64.b64encode(obj_bytes)

def tcp_connect(sock, addr):
    print('Receive a tcp connection from %s:%s...' % addr)
    sock.send(b'Connect successfully! \nWelcome, Client! plz give me your name!')
    first_in = True
    while True:
        data = sock.recv(1024)
        # print(str(data))
        if not data or data.decode('utf-8') == 'exit':
            break
        if first_in:
            sock.send(('Hello, %s! plz enter what you need to encrypt: ' % data.decode('utf-8')).encode('utf-8'))
            first_in = False
        else:
            sock.send('The encrypted text is '.encode('utf-8') + encodeobj(data) + '\nConnection over.'.encode('utf-8'))
    sock.close()
    print('Connection from %s:%s closed.' % addr)


serverName, port = "127.0.0.1", 9891

mSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
mSocket.bind((serverName, port))
mSocket.listen(1)
print("Waiting for client on port "+ str(port) + "...")
sock, addr = mSocket.accept()
tcp_connect(sock, addr)


