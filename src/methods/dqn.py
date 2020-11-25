import torch
import random
import numpy as np
from collections import deque
import cv2
import time
import os
from torch.utils.tensorboard import SummaryWriter

from .base import BaseMethod
from .memories.inmemory_replay import InMemoryReplay
from .nets.cnn import CNN
import vizdoom as vzd
from . import common
from matplotlib import pyplot as plt


class DQN(BaseMethod):

    def build_action(self, a):
        return [1 if a == i else 0 for i in range(self.net.actions)]

    def __init__(self, params=None):
        # self, params, actions, input_shape=(4, 64, 64)):
        self.params = params if params is not None \
            else common.DEFAULT_PARAMS
        self.doom = vzd.DoomGame()
        self.init_doom()
        self.batch_size = self.params['batch_size']
        self.memory = InMemoryReplay(size=self.params['mem_size'], input_shape=self.params['input_shape'])
        self.test_memory = InMemoryReplay(size=self.params['dry_size'], input_shape=self.params['input_shape'])
        self.net = CNN(None, self.doom.get_available_buttons_size())
        self.curr_state = deque(maxlen=self.params['history'])
        self.next_state = deque(maxlen=self.params['history'])
        self.average_qs = []

    def apply_action(self, a):
        frame_skip = self.params['frameskip']
        return self.doom.make_action(a, frame_skip)

    def dry_run(self, steps):
        print(f'Starting a {steps} steps dry-run.')
        curr = 0
        while curr < steps:
            self.doom.new_episode()
            while not self.doom.is_episode_finished():
                # make sure that the state management makes sense
                # confirm in original implementation
                state = self.doom.get_state().screen_buffer
                s = self.state_to_net_state(state, self.curr_state)

                a_ = random.randint(0, self.net.actions)
                # print(a_)
                a = self.build_action(a_)
                r = self.apply_action(a)
                r = self.normalize_reward(r)

                t = self.doom.is_episode_finished()
                if t:
                    next_state = state
                else:
                    next_state = self.doom.get_state().screen_buffer
                s_p = self.state_to_net_state(next_state, self.next_state)
                # print(s_p.shape, s_p)
                # print(s.shape)
                # print(next_state.shape)

                # print(s_p)

                self.test_memory.add_transition(s, a_, s_p, r, t)

                curr += 1
                if curr >= steps:
                    break

            self.curr_state.clear()
            self.next_state.clear()
        avg_q = self.average_q_test()
        print(f'Dry-run finished. Starting avg. Q: {avg_q}')

    def init_doom(self):
        self.doom.load_config(self.params['doom_config'])
        self.doom.init()
        print('Doom initialized.')

    def create_tensorboard(self):
        src_dir = os.environ['VZD_TORCH_DIR']
        scenario = self.params['doom_config'].split('/')[-1].split('.')[0]
        log_path = f'{src_dir}/logs/{scenario}/{self.net.name}/try_{time.time()}'
        os.makedirs(log_path)
        return SummaryWriter(log_dir=log_path)

    def train(self):
        # init doom env
        # load config
        training_steps = 0
        print(f'Training model {self.net.name}. Parameters: {self.net.count_parameters()}.')
        self.dry_run(self.params['dry_size'])
        writer = self.create_tensorboard()
        s, _, _, _, _ = self.test_memory.get_batch(10)
        s = torch.from_numpy(s).to(self.net.device)
        writer.add_graph(self.net, input_to_model=s, verbose=True)
        for episode in range(self.params['episodes']):

            epi_l = 0.
            epi_r = 0.

            self.doom.new_episode()
            while not self.doom.is_episode_finished():
                # make sure that the state management makes sense
                # confirm in original implementation
                state = self.doom.get_state().screen_buffer
                s = self.state_to_net_state(state, self.curr_state)
                # print(s.shape)

                a_ = self.net.next_action(s, training_steps)
                a = self.build_action(a_)
                r = self.apply_action(a)
                r = self.normalize_reward(r)

                t = self.doom.is_episode_finished()
                if t:
                    next_state = state
                else:
                    next_state = self.doom.get_state().screen_buffer

                s_p = self.state_to_net_state(next_state, self.next_state)

                t = self.doom.is_episode_finished()

                self.memory.add_transition(s, a_, s_p, r, t)

                batch = self.memory.get_batch(self.batch_size)
                if not batch:
                    continue

                loss = self.net.train_(batch)
                training_steps += 1

                if training_steps % 10000 == 0:
                    self.serialize_model(training_steps)

                epi_l += loss
                epi_r += r


            avg_q = self.average_q_test()
            self.write_tensorboard(epi_l, epi_r, avg_q)
            self.average_qs.append(avg_q)
            print(f'Episode {episode} ended. Reward earned: {epi_r}. Episode loss: {epi_l}. Avg. Q after episode: {avg_q}')

            self.curr_state.clear()
            self.next_state.clear()

    def state_to_net_state(self, state, d):
        s = self.net.preprocess(state)
        # plt.imshow(s, cmap='gray')
        # plt.show()
        if len(d) == 0:
            [d.append(s) for _ in range(self.net.n_frames)]
            [self.next_state.append(s) for _ in range(self.net.n_frames)]
        else:
            d.append(s)
        # return np.stack(np.array(d))
        return np.array(d)

    def normalize_reward(self, r):
        return r

    def average_q_test(self):
        qs = 0.
        for s in self.test_memory.s:
            s_net = self.net.to_net(s)
            qs += torch.max(self.net.forward(s_net)).cpu().data.numpy()
        qs = qs / self.test_memory.max_size
        return qs

    def serialize_model(self, steps):
        base_dir = os.environ['VZD_TORCH_DIR']
        scenario = self.params['doom_config'].split('/')[-1].split('.')[0]
        scenario_dir = f'{base_dir}/weights/{scenario}'
        os.makedirs(scenario_dir)
        path = f'{scenario_dir}/{self.net.name}_{scenario}_{time.time()}.pt'
        torch.save(self.net.state_dict(), path)
