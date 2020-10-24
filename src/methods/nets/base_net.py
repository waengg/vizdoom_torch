from abc import ABC
from abc import abstractmethod

import torch.nn as nn

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        
    def forward(self, x):
        return None