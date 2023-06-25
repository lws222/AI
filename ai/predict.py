import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd  # 导入pandas模块
import torch.nn.functional as F

df = pd.read_excel('C:/Users/19101/Desktop/shuju.xlsx', usecols=[3, 4, 5, 6, 7, 8, 9, 10])
data = df.loc[90].values
print(data)
data = torch.tensor(data)
data = data.float().to('cuda')

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


model = nnet()
weights_path = "./save_weights/model1.pth"
weights_dict = torch.load(weights_path, map_location='cpu')
model.load_state_dict(weights_dict)
model.to('cuda')
model.eval()
output = model(data)
print(output)