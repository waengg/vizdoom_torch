import torch
from methods.dqn import DQN
from methods.nets.cnn import CNN
import vizdoom as vzd
from collections import deque
import numpy as np

def build_action(a):
    return [1 if a == i else 0 for i in range(3)]

def main():
    net = CNN(None, 3)
    net.load_state_dict(torch.load('/home/gabrielwh/dev/vizdoom_torch/weights/basic_cnn_e1m1_1606417677.3651543.pt'))
    doom = vzd.DoomGame()
    doom.load_config('/home/gabrielwh/dev/vizdoom_torch/configs/e1m1.cfg')
    doom.set_window_visible(True)
    episodes = 100
    doom.init()
    state = deque(maxlen=4)
    for episode in range(episodes):
        doom.new_episode()
        while not doom.is_episode_finished():
            s = doom.get_state().screen_buffer
            s = state_to_net_state(net, s, state)
            a = net.next_action(s)
            doom.make_action(build_action(a))

        state.clear()


def state_to_net_state(net, s, d):
    s = net.preprocess(s)
    # plt.imshow(s, cmap='gray')
    # plt.show()
    if len(d) == 0:
        [d.append(s) for _ in range(4)]
    else:
        d.append(s)
    # return np.stack(np.array(d))
    return np.array(d)

if __name__ == "__main__":
    main()
