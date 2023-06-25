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

    def forward(self, input, hp1, hp2):
        h1 = torch.Tensor(10, 1, 20).to('cuda')
        for i in range(input.size(1)):

            h, hp1 = self.rnn1(input[0][i].view(20, 1, 1), hp1)

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


test_data = 'G:\shiyanmoxing\data/new\data/train/0-300_8_240.CSV'
my_list = list(range(0, 40, 2))
dr = pd.read_csv(test_data, usecols=my_list)
df = pd.read_csv(test_data, usecols=[40, 41, 42, 43, 44])
input = torch.tensor(np.array(dr)).float().view(1, 10, 20).to('cuda')
label = torch.tensor(np.array(df)).float().view(10, 1, 5).to('cuda')
model = rrnn(input1, input2).to('cuda')
weights_path = "./save_weights/rrnn_model_1594.pth"
weights_dict = torch.load(weights_path, map_location='cpu')
model.load_state_dict(weights_dict['model'])
hidden_prev1 = weights_dict['h1'].to('cuda')
hidden_prev2 = weights_dict['h2'].to('cuda')
model.to('cuda')
model.eval()
output,_,_ = model(input,hidden_prev1,hidden_prev2)
print(output)
print(label)
loss_function = nn.MSELoss()
loss = loss_function(output, label)
print(loss)
print(label-output)
