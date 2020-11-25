import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from . import common
from .base_net import BaseNet

# the net has a fixed architecture, but each layer's hyperparameters
# can be altered by passing a dict to this class


class CNN(BaseNet):

    # REMINDER: input dims are BxCxHxW
    def __init__(self, params, actions, input_shape=(4, 64, 64), batch_size=32, gamma=0.9):
        super().__init__(params, actions)
        self.actions = actions
        self.gamma = gamma
        self.name = 'basic_cnn'
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.n_frames = input_shape[0]
        self.anneal_until = 500000
        self.end_eps = 0.1
        self.start_eps = 1.
        self.l = {}  # noqa: E741
        self.l["conv1"] = {
            "w": input_shape[1] - 8 + 1,
            "h": input_shape[2] - 8 + 1,
            "c": 32
        }
        self.l["conv2"] = {
            "w": self.l["conv1"]["w"] - 4 + 1,
            "h": self.l["conv1"]["h"] - 4 + 1,
            "c": 64
        }
        self.l["fc1"] = {
            "f": 128
        }
        self.l["out"] = {
            "f": self.actions
        }
        self.conv1 = nn.Conv2d(input_shape[0], 32, [8, 8])
        self.conv2 = nn.Conv2d(self.l["conv1"]["c"], 64, [4, 4])
        self.fc1 = nn.Linear(self.l["conv2"]["w"] *
                             self.l["conv2"]["h"] *
                             self.l["conv2"]["c"], 128)
        self.out = nn.Linear(self.l["fc1"]["f"], self.actions)

        gpu = common.DEVICE_NUMBER
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available()
                                   else "cpu")
        self.to(self.device)

        self.loss = nn.MSELoss()
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-4)

    def to_net(self, state):
        if state.shape != 4:
            s = np.expand_dims(state, axis=0)
        else:
            s = np.copy(state)
        return s

    def eps(self, steps):
        return self.end_eps if steps >= self.anneal_until else \
            self.start_eps - ((self.start_eps - self.end_eps) /
            self.anneal_until * steps)

    def forward(self, i):
        t = torch.from_numpy(i).type(torch.FloatTensor).to(self.device)
        x = F.relu(self.conv1(t))
        x = F.relu(self.conv2(x))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        out = self.out(x)
        return out

    def preprocess(self, state):
        s = np.array(state)
        s = cv2.resize(s, self.input_shape[1:][::])  # invert input shape
        # return np.expand_dims(s / 255., axis=0)  # add batch dimension
        return s / 255.

    def build_loss(self, qs, qs_p, a, r, t):
        # torch.from_numpy
        return self.future_q(qs, qs_p, a, r, t)
        # return [qs[i] if i != a else r for i in range(self.n_actions)]

    def future_q(self, qs, qs_p, a, r, t):
        f_q = np.copy(qs)
        q = np.argmax(qs_p, axis=1)
        for _i, _t in enumerate(t):
            _r = r[_i]
            _a = a[_i]
            f_q[_i, _a] = _r if _t else _r + q[_i] * self.gamma
        return f_q

    def train(self, batch):
        # memory is built as State x Action x Next State x Reward x Is Terminal
        s, a, s_p, r, t = batch[0], batch[1], batch[2], batch[3], batch[4]
        # s = torch.from_numpy(s)
        # s_p = torch.from_numpy(s_p)
        with torch.no_grad():
            qs_p = self.forward(s_p)
            qs_p_cpu = qs_p.cpu().data.numpy()
        self.optim.zero_grad()
        qs = self.forward(s)
        qs_cpu = qs.cpu().data.numpy()
        f_q = torch.from_numpy(self.build_loss(qs_cpu, qs_p_cpu, a, r, t)).to(self.device)
        # print(f_q)
        loss = self.loss(qs.float(), f_q.float())
        loss.backward()
        self.optim.step()
        return loss.item()

# net = CNN(None, 5)
# t = torch.rand((1, 4, 64, 64))
# print(net.forward(t))
