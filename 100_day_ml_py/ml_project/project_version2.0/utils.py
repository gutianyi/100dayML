import numpy as np
import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from hmmlearn import hmm
from sklearn.preprocessing import OneHotEncoder

# 获取数据
def get_data(fileName,normalize=False):
    # 读取数据
    data = pd.read_csv(fileName)
    # 去除异常值
    data = data.dropna()
    # 归一化
    if (normalize):
        min_data = np.min(data["Adj Close"].values)
        max_data = np.max(data["Adj Close"].values)
        data["Adj Close"] = (data["Adj Close"]-min_data)/(max_data-min_data)

    return data

# 处理文本数据
def get_words():
    # 获取所有文件名
    filepath = "tweet_train\\"
    path_a = []
    path_b = []
    path_c = []
    for i, j, k in os.walk(filepath):
        path_a.append(i);
        path_b.append(j);
        path_c.append(k);

    # 整理文件名列表
    path = []
    for i in range(1, len(path_a)):
        temp = []
        for j in range(len(path_c[i])):
            temp.append(path_a[i] + "\\" + path_c[i][j])
        path.append(temp)

    # tweet数据转换成数据框并保存成csv
    for i in range(len(path)):
        tweet_data = pd.DataFrame()
        for j in range(len(path[i])):
            with open(path[i][j], "r") as f:
                temp = pd.read_json(path[i][j], lines=True)
                temps = [tweet_data, temp]
                tweet_data = pd.concat(temps)
        tweet_data.columns = ["time", "text", "user_id"]
        tweet_data.sort_values("time", inplace=True)
        tweet_data.to_csv(str(i + 1) + "th_tweet_data.csv", index=False)

# 处理文本数据异常字符
def controlStrFormat(temp):
    temp = temp.replace('\'','')
    temp = temp.replace('\"','')
    temp = temp.replace('[','')
    temp = temp.replace(']','')
    temp = temp.replace(',','')
    temp = temp.replace('-','')
    return temp

# 转换文本数据时间
def TransformTime(tweet):
    tweet["time"] = pd.to_datetime(tweet["time"])
    tweet["time"] = tweet["time"].apply(lambda x: x.strftime("%Y/%m/%d"))
    return tweet
# 情感分析
def get_sentiment(fileName):
    # 读取数据
    tweet = pd.read_csv(fileName)
    # 处理text内容
    tweet["text"] = tweet["text"].apply(lambda x: controlStrFormat(x))
    # 调用api情感分析
    temp_list = tweet["text"].tolist()
    analyzer = SentimentIntensityAnalyzer()
    scores_neg = []
    scores_neu = []
    scores_pos = []
    scores_compound = []
    for item in temp_list:
        temp = analyzer.polarity_scores(item)
        scores_neg.append(temp["neg"])
        scores_neu.append(temp["neu"])
        scores_pos.append(temp["pos"])
        scores_compound.append(temp["compound"])
    tweet = pd.concat([tweet, pd.Series(scores_neg).rename("neg"), pd.Series(scores_neu).rename("neu"),
                       pd.Series(scores_pos).rename("pos"), pd.Series(scores_compound).rename("compound")], axis=1)

    # 转换情感值
    tweet["sentiment"] = -1 * tweet["neg"] + 0.5 * tweet["neu"] + tweet["pos"]


    # 转换时间
    tweet = TransformTime(tweet)
    tweet = tweet.groupby("time").mean()
    tweet["sentiment"] = tweet["sentiment"].apply(lambda x: 2 if x >= 0.5 else 1)

    tweet["Date"] = tweet.index
    tweet = tweet.reset_index()
    return tweet[["Date","sentiment"]]

# hmm 预测股市状态
def get_hmm(data,n_states):
    # n is the classes of states
    res = data[["Open", "High", "Low", "Close", "Volume"]].values.tolist()
    model = hmm.GaussianHMM(n_components=n_states, n_iter=2000).fit(res)
    states = model.predict(res)
    # one-hot 编码
    onehotencoder = OneHotEncoder(sparse=False)
    states = onehotencoder.fit_transform(states.reshape(-1, 1))
    columns = ["state_1", "state_2", "state_3", "state_4", "state_5", "state_6", "state_7", "state_8", "state_9",
               "state_10"]
    states_df = pd.DataFrame(states, columns=columns)
    return states_df
# mse
def get_loss(outputs,y):
    return np.mean(np.power(np.array(outputs)-np.array(y),2))

def get_MaxMin(data,target):
    """
    data 数据框
    target 需要归一化的目标列

    """
    min_data = np.min(data[target].values)
    max_data = np.max(data[target].values)
    data[target] = (data[target] - min_data) / (max_data - min_data)
    return data
#if __name__ == "__main__":
#    fileName = "tweet_data/1th_tweet_data.csv"
#    tweet = get_sentiment(fileName)
#    print(tweet)