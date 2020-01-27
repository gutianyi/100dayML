from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from utils import *
from FeatureEngineering import *

# train function
def train(X,y,num_epochs):
    model = LinearRegression()
    for _ in range(num_epochs):
        #训练集和测试集划分
        X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.7,shuffle = False)
        # 训练模型
        model.fit(X_train,y_train)
        outputs = model.predict(X_test)
        #print(outputs)
        loss = get_loss(outputs,y_test)
        print("当前训练次数为{},loss为{:.8f}".format(_+1,loss))

    return model


def predict_nextdays(model,X,tweet_test):
    # extract need value
    result = []
    temp = np.array([X[-1]]+[tweet_test[0]]).reshape(-1,2)
    temp2 = model.predict(temp).reshape(-1)
    result.append(temp2)
    for i in range(6):
        temp = [result[i]]+[tweet_test[i+1]]
        print(temp)
        temp = np.array(temp).reshape(-1,2)
        result.append(model.predict(temp))

    return result

pkl = []
for i in range(8):
    #获取数据
    fileName = "raw_price_train/"+str(i+1)+"_r_price_train.csv"
    fileName_tweet = "tweet_data/"+str(i+1)+"th_tweet_data.csv"
    data = get_data(fileName)

    # 特征工程
    #X,y = feature_engineering_baseline(data,"Adj Close",1,2)
    #print(y)
    #X,y = feature_engineering1(data,"Adj Close",3,2)
    #X,y = feature_engineering2(data)
    X,y = feature_engineering_words(data,fileName_tweet,1,1)
    #X,y = feature_engineering_hmm(data,10,1,1)

    # 模型训练
    num_epochs = 1
    model = train(X,y,num_epochs)

    # 模型预测
    fileName_test = "tweet_test/"+str(i+1)+"th_tweet_test.csv"
    tweet_test = get_test_sentiment(fileName_test)
    predict = predict_nextdays(model,data["Adj Close"].tolist(),tweet_test)

    pkl.append(predict)
    #print(predict)
print(pkl)