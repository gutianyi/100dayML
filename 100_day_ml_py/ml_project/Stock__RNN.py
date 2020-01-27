import torch
import torchvision
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from FeatureEngineering import *
from utils import *


#hardware configuration
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperParameter
input_size = 2
sequence_length = 4
#batch_size = 30
hidden_size = 4
out_size = 2
num_layers = 1
num_epoch = 10000
learning_rate = 0.01


class RNN_stock(nn.Module):

    def __init__(self,input_size,hidden_size,num_layers,out_size):
        super(RNN_stock,self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,dropout=0.3)
        self.fc = nn.Linear(hidden_size,out_size)

    def forward(self,X):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers,X.size(0),self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers,X.size(0),self.hidden_size).to(device)

        #Forward
        #print(h0)
        out,_ = self.lstm(X,(h0,c0))

        #decode hidden state of the last time step
        out = self.fc(out[:,-1,:])

        return out


#get data
fileName = "./raw_price_train/1_r_price_train.csv"
fileName_tweet = "tweet_data/1th_tweet_data.csv"
data = get_data(fileName)

#feature engineering
#X,y = feature_engineering0(data,"Adj Close",sequence_length-1,out_size)
#X,y = feature_engineering1(data,"Adj Close",sequence_length-1,out_size)
#X,y = feature_engineering_words_RNN(data,fileName_tweet,input_size,out_size)
X,y = feature_engineering_hmm_RNN(data,3,input_size,out_size)

X = np.array(X)
y  = np.array(y)

model = RNN_stock(input_size, hidden_size, num_layers, out_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for _ in range(num_epoch):
    X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.9,shuffle=False)
    #X_train = X[:len(X)//2]
    #X_test = X[len(X)//2:]
    #y_train = y[:len(y)//2]
    #y_test = y[len(y)//2:]
    # transform the data
    X_train_tensor = torch.from_numpy(np.array(X_train)).type('torch.FloatTensor')
    y_train_tensor = torch.from_numpy(np.array(y_train)).type('torch.FloatTensor')
    X_test_tensor = torch.from_numpy(np.array(X_test)).type('torch.FloatTensor')
    y_test_tensor = torch.from_numpy(np.array(y_test)).type('torch.FloatTensor')



    X_train_tensor = X_train_tensor.reshape(-1,sequence_length,input_size)
    y_train_tensor = y_train_tensor.reshape(y_train_tensor.size(0),-1)

    out = model(X_train_tensor)
    #print("//////")
    #print(out.data)
    #print("//")
    #print(y_train_tensor)
    loss = criterion(out,y_train_tensor)
    # gradient
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    #print("====")
    #print(model.state_dict())
    if ( _ + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'
              .format( _ + 1, num_epoch, loss.item()))

torch.save(model,"model.pth")

#predicted
print("===================")
model.eval()
mse = []
values = []
with torch.no_grad():
    for i in range(len(X)):
        #X_predicted = torch.from_numpy(np.array(X_res[i])).type('torch.FloatTensor').reshape(-1,sequence_length,input_size)
        X_predicted = torch.from_numpy(np.array(X[i])).type('torch.FloatTensor').reshape(-1, sequence_length,input_size)
        outputs = model(X_predicted)
        #print(outputs.size())
        outputs = outputs.data
        values.append(outputs)
        mse.append(np.power(y[i] - outputs.numpy(),2))
        print(str(outputs)+"//"+str(y[i]))

    #print(mse)
print(np.mean(mse))