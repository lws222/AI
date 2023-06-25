import random

import numpy as np

from matplotlib import pyplot as plt
from torch import optim, nn
import pandas as pd  # 导入pandas模块
import torch.nn.functional as F
from function import *


class nnet(torch.nn.Module):
    def __init__(self):
        super(nnet, self).__init__()
        self.fc1 = torch.nn.Linear(40, 16)
        # self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(16, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.relu(x)
        x = self.fc3(x)
        return x


def train(model, train_filenames, train_dir, lr_scheduler):
    model.train()
    Loss = []
    for xx in train_filenames:
        # print(xx)
        train_datapath = train_dir + xx
        # print(datapath)
        # my_list = list(range(0, 40, 2))
        my_list = list(range(40))
        dr = pd.read_csv(train_datapath, usecols=my_list)
        df = pd.read_csv(train_datapath, usecols=[40, 41, 43, 44])

        input = torch.tensor(np.array(dr)[9]).float().to('cuda')
        label = torch.tensor(np.array(df)[9]).float().to('cuda')
        # print(input.size())
        output = model(input)
        loss = F.mse_loss(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        Loss.append(loss.item())
    return Loss, lr


def eval(model, eval_filenames, eval_dir):
    model.eval()
    Loss = []
    for xx in eval_filenames:
        # print(xx)
        eval_datapath = eval_dir + xx
        # print(datapath)
        # my_list = list(range(0, 40, 2))
        my_list = list(range(40))
        dr = pd.read_csv(eval_datapath, usecols=my_list)
        df = pd.read_csv(eval_datapath, usecols=[40, 41, 43, 44])

        input = torch.tensor(np.array(dr)[9]).float().to('cuda')
        label = torch.tensor(np.array(df)[9]).float().to('cuda')
        # print(input, label)
        output = model(input)
        loss = F.mse_loss(output, label)
        Loss.append(loss.item())
    return Loss


device = 'cuda'
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
model = nnet().to(device)
lr = 0.0001
epochs = 300
optimizer = optim.Adam(model.parameters(), lr, weight_decay=0.01)
lr_scheduler = create_lr_scheduler(optimizer, len(train_filenames), epochs, warmup=True)
Train_Loss = []
Eval_Loss = []
for iter in range(epochs):
    train_loss, lr = train(model, train_filenames, train_dir, lr_scheduler)
    eval_loss = eval(model, val_filenames, val_dir)
    print("Iteration: {} train_loss: {} val_loss: {} lr:{}".format(iter, sum(train_loss) / len(train_loss),
                                                                   sum(eval_loss) / len(eval_loss), lr))
    Eval_Loss.append(sum(eval_loss) / len(eval_loss))
    Train_Loss.append(sum(train_loss) / len(train_loss))
plt.plot(Eval_Loss, 'r', Train_Loss, 'b')
plt.xlabel('训练次数')
plt.ylabel('loss')
plt.title('RNN损失函数下降曲线')
