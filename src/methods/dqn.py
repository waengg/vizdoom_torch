import torch
import numpy as np
from base import BaseMethod
from memories.inmemory_replay import InMemoryReplay
from nets.cnn import CNN
import vizdoom as vzd
import common


class DQN(BaseMethod):

    def __init__(self, params=None):
        # self, params, actions, input_shape=(4, 64, 64)):
        self.doom = vzd.DoomGame()
        self.init_doom()
        self.memory = InMemoryReplay()
        self.net = CNN(None, self.doom.get_available_buttons_size)
        self.params = params if params is not None \
            else common.DEFAULT_PARAMS

    def dry_run(self, steps):
        curr = 0
        while curr < steps:
            self.doom.new_episode()
            while not self.doom.is_episode_finished():
                # make sure that the state management makes sense
                # confirm in original implementation
                state = self.doom.get_state().screen_buffer
                s = self.state_to_net_state(state)

                a = random.randint(0, self.net.actions)
                r = self.apply_action(a)
                r = self.normalize_reward(r)

                s_p = self.doom.get_state().screen_buffer
                s_p = self.state_to_net_state(s_p)

                t = self.doom.is_episode_finished()

                loss = self.net.train()
                self.memory.add_transition(s, a, s_p, r, t)

            self.curr_state.clear()
            self.next_state.clear()

    def init_doom(self):
        self.doom.load_config(self.params['doom_config'])
        self.doom.init()
        print('Doom initialized.')

    def train(self):
        # init doom env
        # load config
        training_steps = 0
        self.init_doom()
        self.doom.init()
        for episode in range(self.params['episodes']):

            epi_l = 0.
            epi_r = 0.

            self.doom.new_episode()
            while not self.doom.is_episode_finished():
                # make sure that the state management makes sense
                # confirm in original implementation
                state = self.doom.get_state().screen_buffer
                s = self.state_to_net_state(state)

                a = self.net.next_action(s, training_steps)
                r = self.apply_action(a)
                r = self.normalize_reward(r)

                s_p = self.doom.get_state().screen_buffer
                s_p = self.state_to_net_state(s_p)

                t = self.doom.is_episode_finished()

                loss = self.net.train()
                self.memory.add_transition(s, a, s_p, r, t)
                epi_l += loss
                epi_r += r

            self.curr_state.clear()
            self.next_state.clear()

    def state_to_net_state(self, state, d):
        s = self.net.preprocess(state)
        if len(d) == 0:
            [d.append(s) for _ in range(self.net.n_frames)]
            [self.next_state.append(s) for _ in range(self.net.n_frames)]
        else:
            d.append(state)
        return np.array(d)

    def normalize_reward(self, r):
        return r
