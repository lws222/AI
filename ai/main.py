import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd  # 导入pandas模块
import torch.nn.functional as F
from tqdm import tqdm

import distributed_utils as utils

dr = pd.read_excel('C:/Users/19101/Desktop/shuju.xlsx', usecols=[1, 2])  # 从指定工作簿中获取数据
df = pd.read_excel('C:/Users/19101/Desktop/shuju.xlsx', usecols=[3, 4, 5, 6, 7, 8, 9, 10])
data = torch.tensor(np.array(df))
target = torch.tensor(np.array(dr))

torch_dataset = TensorDataset(data, target)
# print(len(torch_dataset))
# print(torch_dataset)
train_size = int(len(torch_dataset) * 0.7)
validate_size = len(torch_dataset) - train_size
train_dataset, validate_dataset = torch.utils.data.random_split(torch_dataset,
                                                                [train_size, validate_size])
triandata = DataLoader(
    dataset=train_dataset,  # torch TensorDataset format
    batch_size=4,  # mini batch size
    shuffle=True,  # 要不要打乱数据 (打乱比较好)
    num_workers=0,  # 多线程来读数据
)

valdata = DataLoader(
    dataset=validate_dataset,  # torch TensorDataset format
    batch_size=4,  # mini batch size
    shuffle=True,  # 要不要打乱数据 (打乱比较好)
    num_workers=0,  # 多线程来读数据
)


class nnet(torch.nn.Module):
    def __init__(self):
        super(nnet, self).__init__()
        self.fc1 = torch.nn.Linear(8, 16)
        self.fc2 = torch.nn.Linear(16, 16)
        self.fc3 = torch.nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def train_one(modle, triandata, optimizer,lr_scheduler):
    modle.train()

    m = 0
    Losslist = []
    for ii, (x, y) in enumerate(triandata):
        x, y = x.float().to('cuda'), y.float().to('cuda')
        # print("shuru:", x, "target:", y)
        t = modle(x)
        # print(t)
        loss = F.mse_loss(t, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        Losslist.append(loss.item())

    m = np.mean(Losslist)

    return m,lr

def eval_one(modle, evaldata):
    modle.eval()

    m = 0
    Losslist = []
    for ii, (x, y) in enumerate(evaldata):
        x, y = x.float().to('cuda'), y.float().to('cuda')
        # print("shuru:", x, "target:", y)
        t = modle(x)
        # print(t)
        loss = F.mse_loss(t, y)
        # print(t-y)
        Losslist.append(loss.item())
    m = np.mean(Losslist)

    return m,(t[-1][1]-y[-1][1]).item(), (t[-1][0]-y[-1][0])

epochs = 1000
net1 = nnet().to('cuda')
net1_optimizer = torch.optim.Adam(net1.parameters(), lr=1e-3)
lr_scheduler = create_lr_scheduler(net1_optimizer, len(triandata), epochs, warmup=True)
loss_list1 = []
loss_list2 = []
for i in range(10):
    with tqdm(total=int(epochs / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(epochs / 10)):
            m1, lr = train_one(net1, triandata, net1_optimizer, lr_scheduler)
            m2, cha1, cha2 = eval_one(net1, valdata)
            save_file = net1.state_dict()
            # if epoch//20 == 0:
            #     save_file = {"model": net1.state_dict(),
            #                  "epoch": epoch}
            torch.save(save_file, "save_weights/model1.pth")
            loss_list1.append(m1)
            loss_list2.append(m2)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({'episode': '%d' % (epochs / 10 * i + i_episode + 1),
                                  'trainloss': '%.3f' % m1, 'evalloss': '%.3f' %m2, 'vcha': '%.3f' %cha1,
                                  'xcha': '%.3f' %cha2, 'lr': '%.3f' %lr})
            pbar.update(1)
episodes_list = list(range(len(loss_list1)))
plt.plot(episodes_list, loss_list1)
plt.xlabel('Episodes')
plt.ylabel('loss')
plt.title('trianloss')
plt.show()

episodes_list = list(range(len(loss_list2)))
plt.plot(episodes_list, loss_list2)
plt.xlabel('Episodes')
plt.ylabel('loss')
plt.title('evalloss')
plt.show()

