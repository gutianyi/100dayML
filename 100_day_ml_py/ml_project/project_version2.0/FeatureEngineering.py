import numpy as np
from utils import *
def feature_engineering_baseline(data, target, m,n):
    """
   无特征工程
   X [[adj close]]
   y [adj close]
    """
    # 取出y值
    values = data[target].tolist()
    # 构造X,y
    X = []
    y = []
    for i in range(len(values)-m-n+1):
        temp = values[i:i + m]
        X.append(temp)
        y.append(values[i+m:i+m+n])
    return X, y


def feature_engineering1(data, target, m,n):
    """
    特征工程
    X[[adj close, Volume]]
    y [adj close]
    """
    # 取出y值
    values = data[target].tolist()

    # 标准化Volume
    target = "Volume"
    data = get_MaxMin(data,target)

    # 构造X,y
    X = []
    y = []
    for i in range(len(values) - m-n+1):
        temp = values[i:i + m] + data["Volume"].tolist()[i:i + m]
        X.append(temp)
        y.append(values[i + m:i+m+n])
    return X, y


def feature_engineering2(data, target, m,n):
    # 取出y值
    values = data[target].tolist()

    # 标准化Volume
    target = "Volume"
    data = get_MaxMin(data, target)

    # 标准化y
    target = "Adj Close"
    data = get_MaxMin(data, target)

    # 构造X,y
    X = []
    y = []
    for i in range(len(values) - m-n+1):
        temp = values[i:i + m] + data["Volume"].tolist()[i:i + m]
        X.append(temp)
        y.append(values[i + m:i+m+n])
    return X, y

def feature_engineering_words(data,fileName,m,n):
    """
    fileName 是文本数据csv文件
    """
    values = data["Adj Close"].tolist()
    sentiments = get_sentiment(fileName)
    data = pd.merge(data,sentiments,left_on="Date",right_on="Date")

    # tweet文本处理缺失值
    data["sentiment"] = data["sentiment"].apply(lambda x: 1 if x is None else x )
    # 构建X，y
    X= []
    y = []
    for i in range(len(data)-m-n+1):
        temp = values[i:i + m] + data["sentiment"].tolist()[i:i + m]
        X.append(temp)
        y.append(values[i + m:i+m+n])
    return X,y

def feature_engineering_words_RNN(data,fileName,m,n):
    """
    fileName 是文本数据csv文件
    """
    values = data["Adj Close"].tolist()
    sentiments = get_sentiment(fileName)
    data = pd.merge(data,sentiments,left_on="Date",right_on="Date")

    # tweet文本处理缺失值
    data["sentiment"] = data["sentiment"].apply(lambda x: 1 if x is None else x )
    # 构建X，y
    X= []
    y = []
    for i in range(len(data)-m-n+1):
        temp = values[i:i + m] + data["sentiment"].tolist()[i:i + m]
        X.append(temp)
        y.append(values[i + m:i+m+n])
    X= np.array(X).reshape(-1,m,2)
    y = np.array(y).reshape(-1,n,1)
    return X,y

def feature_engineering_hmm(data,n_states,m,n):

    values = data["Adj Close"].tolist()
    states_df = get_hmm(data,n_states)
    data = pd.concat((data,states_df),axis=1)

    # 构建X，y
    X = []
    y = []
    columns = ["Adj Close"]
    for i in range(n_states):
        columns.append("state_"+str(i+1))

    for i in range(len(data) - m-n+1):
        temp = data[columns].iloc[i:i+m,:].values
        X.append(temp)
        y.append(values[i + m:i + m + n])
    X = np.array(X).reshape(-1,n_states+1)
    y = np.array(y)
    return X,y

def feature_engineering_hmm_RNN(data,n_states,m,n):

    values = data["Adj Close"].tolist()
    states_df = get_hmm(data,n_states)
    data = pd.concat((data,states_df),axis=1)

    # 构建X，y
    X = []
    y = []
    columns = ["Adj Close"]
    for i in range(n_states):
        columns.append("state_"+str(i+1))

    for i in range(len(data) - m-n+1):
        temp = data[columns].iloc[i:i+m,:].values
        X.append(temp)
        y.append(values[i + m:i + m + n])
    X = np.array(X).reshape(-1,m,n_states+1)
    y = np.array(y).reshape(-1,n,1)

    return X,y
def get_test_sentiment(fileName):

    tweet_test = get_sentiment(fileName)
    #print(tweet_test)
    temp = tweet_test["sentiment"].tolist()
    res = temp[:4] + temp[7:10]
    return res

if __name__ == "__main__":
#    fileName1 = "raw_price_train/1_r_price_train.csv"
#    fileName2 = "tweet_data/1th_tweet_data.csv"
#    data = get_data(fileName1)
#    X,y = feature_engineering_hmm(data,10,1,1)
#    print(X[0])
    fileName = "tweet_test/1th_tweet_test.csv"
    print(get_test_sentiment(fileName))