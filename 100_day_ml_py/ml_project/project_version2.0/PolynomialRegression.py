from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from utils import *
from FeatureEngineering import *
from sklearn.preprocessing import PolynomialFeatures

# train function
def train(X,y,degree,num_epochs):
    # 映射X
    qf = PolynomialFeatures(degree=degree)
    X_qf = qf.fit_transform(X)

    model = LinearRegression()
    for _ in range(num_epochs):
        #训练集和测试集划分
        X_train,X_test,y_train,y_test = train_test_split(X_qf,y,train_size = 0.7,shuffle = False)
        # 训练模型
        model.fit(X_train,y_train)
        outputs = model.predict(X_test)
        #print(outputs)
        loss = get_loss(outputs,y_test)
        print("当前训练次数为{},loss为{:.8f}".format(_+1,loss))

    return model

def predict_nextdays(model,data):
    return

#获取数据
fileName = "raw_price_train/1_r_price_train.csv"
fileName_tweet = "tweet_data/1th_tweet_data.csv"
data = get_data(fileName)

# 特征工程
# 特征工程
#X,y = feature_engineering0(data,"Adj Close",1,1)
#X,y = feature_engineering1(data,"Adj Close",1,1)
#X,y = feature_engineering2(data)
#X,y = feature_engineering_words(data,fileName_tweet,1,1)
X,y = feature_engineering_hmm(data,3,1,1)

# 模型训练
num_epochs = 10
model = train(X,y,3,num_epochs)

