from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from utils import *
from FeatureEngineering import *
# train function
def train(X,y,num_epochs):
    for _ in range(num_epochs):
        #训练集和测试集划分
        X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.7,shuffle = False)
        # 训练模型
        model = LinearRegression()
        model.fit(X_train,y_train)
        outputs = model.predict(X_test)
        loss = get_loss(outputs,y_test)
        print("当前训练次数为{},loss为{:.8f}".format(_+1,loss))

#获取数据
fileName = "raw_price_train/1_r_price_train.csv"
data = get_data(fileName)

# 特征工程
X,y = feature_engineering0(data)

# 模型训练
num_epochs = 100
train(X,y,num_epochs)
