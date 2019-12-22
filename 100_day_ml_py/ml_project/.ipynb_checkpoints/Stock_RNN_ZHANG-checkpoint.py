import torch
import torchvision
import torch.nn as nn
import numpy as np
import pandas as pd
from preprocess import preprocess
from sklearn.model_selection import train_test_split

#hardware configuration
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperParameter
input_size = 1
sequence_length = 1
batch_size = 100
hidden_size = 4
out_size = 1
num_layers = 1
num_epoch = 1000
learning_rate = 0.4


class RNN_stock(nn.Module):

    def __init__(self,input_size,hidden_size,num_layers,out_size):
        super(RNN_stock,self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size,out_size)

    def forward(self,X):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers,X.size(0),self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers,X.size(0),self.hidden_size).to(device)

        #Forward
        #print(X.size(0))
        out,_ = self.lstm(X,(h0,c0))

        #decode hidden state of the last time step
        out = self.fc(out[:,-1,:])

        return out

#preprocess
filepath = "./raw_price_train/1_r_price_train.csv"
data = pd.read_csv(filepath)
X,y = preprocess(data,sequence_length)


model = RNN_stock(input_size, hidden_size, num_layers, out_size)
model.train()
for _ in range(num_epoch):
    X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7)

    # transform the data
    X_train_tensor = torch.from_numpy(np.array(X_train)).type('torch.FloatTensor')
    y_train_tensor = torch.from_numpy(np.array(y_train)).type('torch.FloatTensor')
    X_test_tensor = torch.from_numpy(np.array(X_test)).type('torch.FloatTensor')
    y_test_tensor = torch.from_numpy(np.array(y_test)).type('torch.FloatTensor')

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_train_tensor = X_train_tensor.reshape(-1,sequence_length,input_size)
    y_train_tensor = y_train_tensor.reshape(y_train_tensor.size(0),-1)

    out = model(X_train_tensor)
    loss = criterion(out,y_train_tensor)

    # gradient
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if ( _ + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'
              .format( _ + 1, num_epoch, loss.item()))

#predicted
model.eval()
with torch.no_grad():
    mse = []
    for i in range(10):
        index = np.random.randint(0,len(X))
        X_predicted = torch.from_numpy(np.array(X[index])).type('torch.FloatTensor').reshape(-1,sequence_length,input_size)
        outputs = model(X_predicted)
    #print(outputs.size())
        outputs = outputs.data
        mse.append(np.power(y[index]-outputs,2))

    print(np.mean(mse))