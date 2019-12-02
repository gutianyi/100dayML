import socket 

client = socket.socket()
host = socket.gethostname()
port = 1889
client.connect((host, port))
while True:
    server_reply = str(client.recv(1024), encoding = "UTF-8")
    print("Server : {}".format(server_reply ))
    msg = input("Client :")
    client.send(bytes(msg, encoding = "UTF-8"))
    while True:
        server_reply = str(client.recv(1024), encoding = "UTF-8")
        print("Server : {}".format(server_reply ))
        msg = input("Client :")
        client.send(bytes(msg, encoding = "UTF-8"))   