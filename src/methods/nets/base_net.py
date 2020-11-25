import torch
import torch.nn as nn
import numpy as np
from random import random, randint
from abc import ABC, abstractmethod


class BaseNet(nn.Module, ABC):
    def __init__(self, params):
        self.params = params
        self.n_actions = self.params['n_actions']
        super(BaseNet, self).__init__()

    @abstractmethod
    def forward(self, x):
        pass

    def eps(self, steps):
        return 0.01

    @abstractmethod
    def preprocess(self, state):
        pass

    def next_action(self, state, steps=None):
        """
        returns the next best action
        defined by argmax(self.net.forward())
        state: an ndarray of the current state, preprocessed
        returns: the index of the best action
        """
        if steps:
            if random() <= self.eps(steps):
                return randint(0, self.n_actions-1)
        qs = self.forward(state)
        return torch.max(qs, axis=1).data.numpy()[0]

    @abstractmethod
    def train(self, batch):
        pass
