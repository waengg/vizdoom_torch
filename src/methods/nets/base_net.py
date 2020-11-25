import torch
import torch.nn as nn
import numpy as np
from random import random, randint
from abc import ABC, abstractmethod


class BaseNet(nn.Module, ABC):
    def __init__(self, params, actions):
        self.params = params
        self.n_actions = actions
        super(BaseNet, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

    def eps(self, steps):
        return 0.01

    @abstractmethod
    def preprocess(self, state):
        pass

    @abstractmethod
    def to_net(self, state):
        pass

    def next_action(self, state, steps=None):
        """
        returns the next best action
        defined by argmax(self.net.forward())
        state: an ndarray of the current state, preprocessed
        returns: the index of the best action
        """
        if steps:
            eps = self.eps(steps)
            if random() <= eps:
                return randint(0, self.n_actions-1)
        s_net = self.to_net(state)
        qs = self.forward(s_net)
        return torch.max(qs, axis=1)[1].cpu().numpy()[0]

    @abstractmethod
    def train(self, batch):
        pass

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
