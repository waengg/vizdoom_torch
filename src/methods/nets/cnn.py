import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import common
from base_net import BaseNet

# the net has a fixed architecture, but each layer's hyperparameters
# can be altered by passing a dict to this class


class CNN(BaseNet):

    # REMINDER: input dims are BxCxHxW
    def __init__(self, params, actions, input_shape=(4, 64, 64)):
        super(CNN, self).__init__(params)
        self.actions = actions
        self.n_frames = input_shape[0]
        self.l = {}  # noqa: E741
        self.l["conv1"] = {
            "layer": nn.Conv2d(input_shape[0], 32, [8, 8]),
            "w": input_shape[1] - 8 + 1,
            "h": input_shape[2] - 8 + 1,
            "c": 32
        }
        self.l["conv2"] = {
            "layer": nn.Conv2d(self.l["conv1"]["c"], 64, [4, 4]),
            "w": self.l["conv1"]["w"] - 4 + 1,
            "h": self.l["conv1"]["h"] - 4 + 1,
            "c": 64
        }
        self.l["fc1"] = {
            "layer": nn.Linear(self.l["conv2"]["w"] *
                               self.l["conv2"]["h"] *
                               self.l["conv2"]["c"], 128),
            "f": 128
        }
        self.l["out"] = {
            "layer": nn.Linear(self.l["fc1"]["f"], self.actions),
            "f": self.actions
        }

        self.loss = nn.MSELoss()
        self.optim = torch.optim.Adam(self.parameters())
        gpu = common.DEVICE_NUMBER
        self.to(torch.device(f"cuda:{gpu}" if torch.cuda.is_available()
                             else "cpu"))

    def forward(self, i):
        x = F.relu(self.l["conv1"]["layer"](i))
        x = F.relu(self.l["conv2"]["layer"](x))
        x = x.flatten(1)
        x = F.relu(self.l["fc1"]["layer"](x))
        print(x.shape)
        out = self.l["out"]["layer"](x)
        return out

    def preprocess(self, state):
        s = np.array(state)
        s = cv2.resize(s, self.input_shape[1:][::])  # invert input shape
        return np.expand_dims(s / 255., axis=0)  # add batch dimension

    def build_loss(self, qs, a, r, t):
        # torch.from_numpy
        i = np.stack(np.arange(0, self.batch_size), a).astype('int32')
        qs[i] = self.future_q(qs, i, r, t)
        return [qs[i] if i != a else r for i in range(self.n_actions)]

    def future_q(qs, i, r, t):
        _qs = np.copy(qs)
        for _i, _t in enumerate(t):
            q = _qs[i[1, _i]
            _qs[i[1, _i] = r if _t else r + q * self.gamma
        return _qs

    def train(self, batch):
        # memory is built as State x Action x Next State x Reward x Is Terminal
        s, a, s_p, r, t = list(zip(*batch))
        s = torch.from_numpy(s)
        s_p = torch.from_numpy(s_p)
        with torch.no_grad():
            qs_p = self.forward(s_p)
        self.optim.zero_grad()
        qs = self.forward(s)
        qs_p = self.build_loss(qs, a, t)
        self.loss(qs, qs_p)
        self.loss.backward()
        self.optim.step()
        return self.loss.item()

# net = CNN(None, 5)
# t = torch.rand((1, 4, 64, 64))
# print(net.forward(t))
