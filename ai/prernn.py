import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
import convlstm

model = convlstm.ConvLSTM(1, 1, (1, 5),1)
train_dir = 'G:/shiyanmoxing/data/data_wait_update/data_update/train/'
val_dir = 'G:/shiyanmoxing/data/data_wait_update/data_update/val/'
train_filename = train_dir + 'LIST.txt'
val_filename = val_dir + 'LIST.txt'
f1 = open(train_filename, "r", encoding='utf-8')
train_filenames = f1.read().splitlines()
f2 = open(val_filename, "r", encoding='utf-8')
val_filenames = f2.read().splitlines()
# print(len(filenames))
lr = 0.001
optimizer = optim.Adam(model.parameters(), lr)
l = []

hidden_prev = []
for iter in range(3000):
    Loss = []
    for xx in train_filenames:
        # print(xx)
        train_datapath = train_dir + xx
        # print(datapath)
        my_list = list(range(0, 40, 2))

        dr = pd.read_csv(train_datapath, usecols=my_list)
        df = pd.read_csv(train_datapath, usecols=[ 42, 43, 44,45,46])

        input = torch.tensor(np.array(dr)).float().view(10, 1, 1, 1, 20).to('cuda')
        label = torch.tensor(np.array(df)).float().view(1, 10, 5).to('cuda')

        output, hidden_prev = model(input, None)

        output = output[0].float().view(1, 10, 5).to('cuda')
        # print(output)
        # print(label)
        loss_function = nn.MSELoss()
        loss = loss_function(output, label)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        Loss.append(loss.item())
    # if iter % 100 == 0:
    print("Iteration: {} loss {}".format(iter, sum(Loss)/len(Loss)))
    l.append(sum(Loss)/len(Loss))

        ##############################绘制损失函数#################################
    print(1)
plt.plot(l, 'r')
plt.xlabel('训练次数')
plt.ylabel('loss')
plt.title('RNN损失函数下降曲线')

