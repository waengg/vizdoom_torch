import torch
import torch.nn as nn
import torch.nn.functional as F
from base_net import BaseNet

# the net has a fixed architecture, but each layer's hyperparameters
# can be altered by passing a dict to this class


class CNN(BaseNet):

    def __init__(self, params, actions, input_shape=(4, 64, 64)):
        super(CNN, self).__init__()
        self.actions = actions
        self.n_frames = input_shape[0]
        self.l = {}
        self.l["conv1"] = {
            "layer": nn.Conv2d(input_shape[0], 32, [8,8]),
            "w": input_shape[1] - 8 + 1,
            "h": input_shape[2] - 8 + 1,
            "c": 32
        }
        self.l["conv2"] = {
            "layer": nn.Conv2d(self.l["conv1"]["c"], 64, [4,4]),
            "w": self.l["conv1"]["w"] - 4 + 1,
            "h": self.l["conv1"]["h"] - 4 + 1,
            "c": 64
        }
        self.l["fc1"] = {
            "layer": nn.Linear(self.l["conv2"]["w"] * self.l["conv2"]["h"] * self.l["conv2"]["c"], 128),
            "f": 128
        }
        self.l["out"] = {
            "layer": nn.Linear(self.l["fc1"]["f"], self.actions),
            "f": self.actions
        }

    def forward(self, i):
        x = F.relu(self.l["conv1"]["layer"](i))
        print(x.shape)
        x = F.relu(self.l["conv2"]["layer"](x))
        print(x.shape)
        x = x.flatten(1)
        print(x.shape)
        print(self.l["fc1"]["layer"].in_features, self.l["conv2"]["h"], self.l["conv2"]["w"], self.l["conv2"]["c"])
        x = F.relu(self.l["fc1"]["layer"](x))
        print(x.shape)
        q = self.l["out"]["layer"](x)
        return q

    def loss(self, q):
        pass


# net = CNN(None, 5)
# t = torch.rand((1, 4, 64, 64))
# print(net.forward(t))