import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_net import BaseNet


class RND(BaseNet):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, [3, 3], stride=2)
        self.conv2 = nn.Conv2d(32, 64, [3, 3], stride=2)
        self.conv3 = nn.Conv2d(64, 96, [3, 3], stride=2)
        self.fc1 = nn.Linear(7*7*96, 128)
        self.out = nn.Linear(128, 96)
        gpu = 0
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available()
                                   else "cpu")
        self.to(self.device)
        self.loss = nn.MSELoss()
        self.optim = torch.optim.Adam(self.parameters())

    def forward(self, i):
        x = F.relu(self.conv1(i))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.out(x)

    def to_net(self, state):
        if len(state.shape) != 4:
            s = np.expand_dims(state, axis=0)
        else:
            s = np.copy(state)
        t = torch.from_numpy(s).type(torch.FloatTensor).to(self.device)
        return t

rnd = RND()
t = np.zeros((32,4,64,64))
t = torch.from_numpy(t).type(torch.FloatTensor).to(rnd.device)
print(rnd.forward(t))