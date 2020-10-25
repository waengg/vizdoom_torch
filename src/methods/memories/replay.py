from abc import ABC, abstractmethod


class classname(ABC):

    def __init__(self, size):
        pass

    @abstractmethod
    def store(self, x):
        pass

    @abstractmethod
    def get_batch(self, b):
        pass
