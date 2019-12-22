# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# %%
# 读取数据集
data = pd.read_csv("raw_price_train/1_r_price_train.csv")
data.head()


# %%
data

# %% [markdown]
#  # 数据预处理

# %%
def preprocess(data,m):
    '''
    data: the dataframe of stock price
    m: the length of sequence
    '''
    adj_close = data["Adj Close"].tolist()
    #
    res_X = []
    res_y = []
    # 剔除前4个元素
    for i in range(4,len(adj_close)-m):
        res_X.append(adj_close[i:i+m])
        res_y.append(adj_close[i+m])
    return res_X,res_y


# %%
# 处理数据 m为子序列长度
m = 5
X,y = preprocess(data,m)


# %%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)


# %%
# 训练模型
model = LinearRegression()
model.fit(X_train,y_train)


# %%
model.score(X_test,y_test)


# %%
y_est = model.predict(X_test)


# %%
np.power(y_test-y_est,2).mean()


# %%
import matplotlib.pyplot as plt 


# %%
plt.plot(np.power(y_test-y_est,2))


# %%


