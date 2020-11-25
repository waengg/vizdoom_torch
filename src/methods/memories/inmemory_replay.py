import numpy as np

from replay import Replay


class InMemoryReplay(Replay):
    def __init__(self, size, input_shape):
        shape = size + input_shape
        # data structure: State x Action x Next State x Reward x Terminal
        self.s = np.zeros(shape, dtpye='float32')
        self.s_p = np.zeros(shape, dtpye='float32')
        self.a = np.zeros(size, dtpye='int32')
        self.r = np.zeros(size, dtpye='float32')
        self.t = np.zeros(size, dtpye='float32')
        self.curr = 0
        self.max_size = size

    def add_transition(self, s, a, s_p, r, t):
        self.s[self.curr] = s
        self.a[self.curr] = a
        self.s_p[self.curr] = s_p
        self.r[self.curr] = r
        self.t[self.curr] = t
        self.curr = min(self.curr + 1, self.max_size)

    def get_batch(self, batch_size):
        i = np.random.randint(0, self.curr, size=(batch_size,))
        return self.s[i], self.a[i], self.s_p[i], self.r[i], self.t[i]
