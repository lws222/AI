import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim

input1 = [1, 64, 2, False]
input2 = [20, 64, 2, False]
class rrnn(nn.Module):
    def __init__(self, input1, input2):
        super(rrnn, self).__init__()
        self.input_size1 = input1[0]
        self.hidden_size1 = input1[1]
        self.num_layers1 = input1[2]
        self.batch_first1 = input1[3]
        self.input_size2 = input2[0]
        self.hidden_size2 = input2[1]
        self.num_layers2 = input2[2]
        self.batch_first2 = input2[3]
        self.rnn1 = nn.RNN(
            input_size=self.input_size1,
            hidden_size=self.hidden_size1,
            num_layers=self.num_layers1,
            batch_first=self.batch_first1,
        )
        self.rnn2 = nn.RNN(
            input_size=self.input_size2,
            hidden_size=self.hidden_size2,
            num_layers=self.num_layers2,
            batch_first=self.batch_first2,
        )
        self.linear1 = nn.Linear(64, 1)
        self.linear2 = nn.Linear(20, 5)
        self.linear3 = nn.Linear(64, 5)
        self.relu = nn.ReLU()
    def forward(self, input, hp1, hp2):
        h1 = torch.Tensor(10, 1, 20).to('cuda')
        for i in range(input.size(1)):

            h, hp1 = self.rnn1(input[0][i].view(20, 1, 1), hp1)
            h = self.relu(h)
            h = self.linear1(h)
            # print(h.size())
            # ss
            # h = self.linear2(h.view(1, 20))

            h1[i] = h.view(1, 20)
        # h1 = torch.tensor(h1)
        # print(h1.size())
        # h2 = self.linear(h1)
        h3, hp2 = self.rnn2(h1, hp2)
        h3 = self.linear3(h3)
        return h3, hp1, hp2

def train(model, train_filenames, train_dir, h1 ,h2):
    model.train()
    Loss = []
    for xx in train_filenames:
        # print(xx)
        train_datapath = train_dir + xx
        # print(datapath)
        my_list = list(range(0, 40, 2))

        dr = pd.read_csv(train_datapath, usecols=my_list)
        df = pd.read_csv(train_datapath, usecols=[40, 41, 42, 43, 44])

        input = torch.tensor(np.array(dr)).float().view(1, 10, 20).to('cuda')
        label = torch.tensor(np.array(df)).float().view(10, 5).to('cuda')

        output, h1, h2 = model(input, h1, h2)

        h1 = h1.detach()
        h2 = h2.detach()
        # print(output.squeeze(), output)
        # ss
        output = output.squeeze().float().to('cuda')
        # loss_function = nn.MSELoss
        # loss_function = nn.CrossEntropyLoss()
        # loss = loss_function(nn.functional.softmax(output, dim=1), label)
        loss_function = nn.L1Loss()
        loss = loss_function(output, label)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        Loss.append(loss.item())
    return Loss, h1, h2

def val(model, val_filenames, val_dir,h1,h2):
    model.eval()
    l = []
    Loss = []
    for xx in val_filenames:
        # print(xx)
        train_datapath = val_dir + xx
        # print(datapath)
        my_list = list(range(0, 40, 2))
        dr = pd.read_csv(train_datapath, usecols=my_list)
        df = pd.read_csv(train_datapath, usecols=[40, 41, 42, 43, 44])
        input = torch.tensor(np.array(dr)).float().view(1, 10, 20).to('cuda')
        label = torch.tensor(np.array(df)).float().view(10, 5).to('cuda')
        # print(input,label)
        output, h1, h2 = model(input, h1, h2)
        # print(output)
        # h1 = h1.detach()
        # h2 = h2.detach()
        output = output.squeeze().float().to('cuda')
        # loss_function = nn.CrossEntropyLoss()
        # loss = loss_function(nn.functional.softmax(output, dim=1), label)
        loss_function = nn.L1Loss()
        loss = loss_function(output, label)
        Loss.append(loss.item())
    return Loss


model = rrnn(input1, input2).to('cuda')
train_dir = 'G:/shiyanmoxing/data/new/data/train/'
val_dir = 'G:/shiyanmoxing/data/new/data/val/'
train_filename = train_dir + 'LIST.txt'
val_filename = val_dir + 'LIST.txt'
f1 = open(train_filename, "r", encoding='utf-8')
train_filenames = f1.read().splitlines()
f2 = open(val_filename, "r", encoding='utf-8')
val_filenames = f2.read().splitlines()
random.shuffle(train_filenames)
random.shuffle(val_filenames)
# print(len(filenames))
lr = 0.1
optimizer = optim.Adam(model.parameters(), lr,weight_decay=0.01)
hidden_prev1 = torch.zeros(2, 1, 64).to('cuda')
hidden_prev2 = torch.zeros(2, 1, 64).to('cuda')
l =[]
for iter in range(3000):
    train_loss, hidden_prev1, hidden_prev2 = train(model, train_filenames, train_dir,hidden_prev1, hidden_prev2)
    val_loss = val(model, val_filenames, val_dir,hidden_prev1, hidden_prev2)
    # if iter % 100 == 0:
    print("Iteration: {} train_loss: {} val_loss: {}".format(iter, sum(train_loss)/len(train_loss), sum(val_loss)/len(val_loss)))
    l.append(sum(val_loss)/len(val_loss))

##############################绘制损失函数#################################
    print(1)
plt.plot(l, 'r')
plt.xlabel('训练次数')
plt.ylabel('loss')
plt.title('RNN损失函数下降曲线')