import time
import socket
from random import choice

from urllib import request,parse
import urllib
import http.cookiejar
from bs4 import BeautifulSoup


#导入夸夸语料对
q_list = []
a_list = []
with open('kuakua_qa_pairs.txt', 'r', encoding='utf-8') as inf:
    for line in inf:
        items = line.strip('\r\n').split('\t')
        q_list.append(items[0])
        a_list.append(items[1])    
        
#生成字典
kuakua_dict = {}
for i in range(len(q_list)):
    if (q_list[i] in kuakua_dict) == False:
        kuakua_dict[q_list[i]] = []
        kuakua_dict[q_list[i]].append(a_list[i])
    if (q_list[i] in kuakua_dict) == True:
        kuakua_dict[q_list[i]].append(a_list[i])

        
#导入停用词表
stop_word_file = 'stop_ch.txt'
stopwords = [line.strip() for line in open(stop_word_file, encoding='UTF-8').readlines()]

#问句语料切句
import jieba
q_set = list(set(q_list))
q_jieba_list = []
for i in range(len(q_set)):
    temp = list(jieba.cut(q_set[i]))
    temp = [x for x in temp if (x in stopwords) == False]
    q_jieba_list.append(temp)

#导入词向量模型
from gensim.models import Word2Vec
model = Word2Vec.load("/Users/elvis/ITProjects/GitHub/PythonTask/100dayML/100_day_ml_py/internet/kuakua_netbot/model/word2vec_kuakua.model")

#WMD距离匹配初始化
from gensim.similarities import WmdSimilarity
num_best = 1
instance = WmdSimilarity(q_jieba_list , model, num_best)

server = socket.socket()
host = socket.gethostname()
port = 1889
server.bind((host, port))

server.listen(5)
while True:
    print("Waiting for client to conect")
    client, addr = server.accept()
    print("Clinet connected, the address is {}".format(addr))
    client.send(bytes("你好，我是金圣水傻逼夸夸机器人。请问你是谁？", encoding = "UTF-8"))
    while True:
        reply = str(client.recv(1024), encoding = "UTF-8")
        re_reply_msg = "你好 " + reply + "，你可真是个傻逼。"  + "快让我抹了蜜的小嘴夸夸你吧"
        client.send(bytes(re_reply_msg, encoding = "UTF-8"))
        while True:
            reply = str(client.recv(1024), encoding = "UTF-8")
            print("Client : {}".format(reply))
            #对客户REPLY做的操作，WMD距离的匹配
            query = list(jieba.cut(reply))
            query = [x for x in query if (x in stopwords) == False]
            sims = instance[query]
            if sims != []:
                candidate_list = kuakua_dict[q_set[sims[0][0]]]
                re_reply_msg = choice(candidate_list)
                client.send(bytes(re_reply_msg, encoding = "UTF-8"))
            else:
                url = "http://www.baidu.com/s?wd=" + urllib.parse.quote(reply)

                headers={"Accept": "text/html, application/xhtml+xml, image/jxr, */*",

                         "Accept - Encoding": "gzip, deflate, br",

                         "Accept - Language": "zh - CN",

                         "Connection": "Keep - Alive",
                         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 Edge/16.16299",
                         "referer":"baidu.com"}
                cookie_jar = http.cookiejar.CookieJar()
                opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
                headall=[]
                for key,value in headers.items():
                    item=(key,value)
                    headall.append(item)
                    opener.addheaders=headall
                    urllib.request.install_opener(opener)
                    data = urllib.request.urlopen(url).read().decode('utf-8')
                    soup = BeautifulSoup(data,'html.parser')
                    time.sleep(10)
                if soup.find_all('h3',class_='t') != []:
                    title_1 = soup.find_all('h3',class_='t')[0].find("a").get_text()
                    url_1 = soup.find_all('h3',class_='t')[0].find("a").get("href")
                    title_2 = soup.find_all('h3',class_='t')[4].find("a").get_text()
                    url_2 = soup.find_all('h3',class_='t')[4].find("a").get("href")
                    re_reply_msg = "让我来帮你求助于搜索引擎呀~" + "\n" +"connecting..." + "\n" + title_1 + "\n" + url_1 + "\n" + title_2 + "\n" + url_2
                    client.send(bytes(re_reply_msg, encoding = "UTF-8"))
                else:
                    headall=[]
                    for key,value in headers.items():
                        item=(key,value)
                        headall.append(item)
                        opener.addheaders=headall
                        urllib.request.install_opener(opener)
                        data = urllib.request.urlopen(url).read().decode('utf-8')
                        soup = BeautifulSoup(data,'html.parser')
                        time.sleep(10)
                    if soup.find_all('h3',class_='t') != []:
                        title_1 = soup.find_all('h3',class_='t')[0].find("a").get_text()
                        url_1 = soup.find_all('h3',class_='t')[0].find("a").get("href")
                        title_2 = soup.find_all('h3',class_='t')[0].find("a").get_text()
                        url_2 = soup.find_all('h3',class_='t')[0].find("a").get("href")
                        re_reply_msg = "让我来帮你求助于搜索引擎呀~" + "\n" +"connecting..." + "\n" + title_1 + "\n" + url_1 + "\n" + title_2 + "\n" + url_2
                        client.send(bytes(re_reply_msg, encoding = "UTF-8"))
                    else:
                        re_reply_msg = "sorry connecting failed"
                        client.send(bytes(re_reply_msg, encoding = "UTF-8"))
                            
                    
                

        
    client.close()