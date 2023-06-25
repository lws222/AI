
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
# rnn = nn.RNN(40, 10, 2)
# h_0 = torch.randn(2, 10, 10)
# seq = torch.randn(1, 10, 40)
# # print(seq)
# output, h = rnn(seq, h_0)
# print(output.size())


input_size = 20
hidden_size = 16
output_size = 5
num_layers = 1


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Net, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
        )
        for p in self.rnn.parameters():
            nn.init.normal_(p, mean=0.0, std=0.001)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)
        # [b, seq, h]
        out = out.view(-1, hidden_size)
        out = self.linear(out)  # [seq,h] => [seq,3]
        nn.Dropout()
        out = out.unsqueeze(dim=0)  # => [1,seq,3]
        return out, hidden_prev

def train(model, filenames, dir, hidden_prev, optimizer):
    model.train()
    for x in filenames:
        datapath = dir+x

        # my_list = list(range(40))
        my_list = list(range(0, 40, 2))
        dr = pd.read_csv(datapath, usecols=my_list)
        df = pd.read_csv(datapath, usecols=[40, 41, 42, 43, 44])

        x = torch.tensor(np.array(dr)).float().view(1, 10, 20).to('cuda')
        y = torch.tensor(np.array(df)).float().view(1, 10, 5).to('cuda')


        # print(hidden_prev.size())
        output, hidden_prev = model(x, hidden_prev)
        hidden_prev =hidden_prev.detach()
        # print(hidden_prev.size())
        loss_function = nn.MSELoss()
        loss = loss_function(output, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, hidden_prev

def val(model, filenames, dir, hidden_prev):
    model.eval()
    for x in filenames:
        datapath = dir + x

        # my_list = list(range(40))
        my_list = list(range(0, 40, 2))
        dr = pd.read_csv(datapath, usecols=my_list)
        df = pd.read_csv(datapath, usecols=[40, 41, 42, 43, 44])

        x = torch.tensor(np.array(dr)).float().view(1, 10, 20).to('cuda')
        y = torch.tensor(np.array(df)).float().view(1, 10, 5).to('cuda')

        # print(hidden_prev.size())
        output, hidden_prev = model(x, hidden_prev)
        hidden_prev = hidden_prev.detach()
        # print(hidden_prev.size())
        loss_function = nn.MSELoss()
        loss = loss_function(output, y)
        return loss

model = Net(input_size, hidden_size, num_layers).to('cuda')
lr = 0.01
train_dir = 'G:/shiyanmoxing/data/new/data/train/'
val_dir = 'G:/shiyanmoxing/data/new/data/val/'
train_filename = train_dir + 'LIST.txt'
val_filename = val_dir + 'LIST.txt'
f1 = open(train_filename, "r", encoding='utf-8')
train_filenames = f1.read().splitlines()
f2 = open(val_filename, "r", encoding='utf-8')
val_filenames = f2.read().splitlines()
# print(len(filenames))
optimizer = optim.Adam(model.parameters(), lr, weight_decay=1)
hidden_prev = torch.zeros(num_layers, 10, hidden_size).to('cuda')
l =[]
for iter in range(300000):
    train_loss, hidden_prev = train(model, train_filenames, train_dir, hidden_prev, optimizer)
    val_loss = val(model, val_filenames, val_dir, hidden_prev)

    if iter % 100 == 0:
        print("Iteration: {} train_loss {} eval_loss {}".format(iter, train_loss.item(), val_loss.item()))
        l.append(train_loss.item())
        ##############################绘制损失函数#################################
    plt.plot(l, 'r')
    plt.xlabel('训练次数')
    plt.ylabel('loss')
    plt.title('RNN损失函数下降曲线')

