#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque, namedtuple
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import math
from itertools import count
import gc
from curious_agent.agents import Agent
from curious_agent.models.cnn_model import CNNModel
import time
from torch.autograd import Variable
import json
import uuid
from curious_agent.environments.open_ai.atari_environments.open_ai_environment import AtariEnvironment
from curious_agent.buffers import PrioritizedBuffer, ReplayBuffer
from curious_agent.meta.default_meta import DefaultMetaData


torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class DQNAgent(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example:
            paramters for neural network
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(DQNAgent, self).__init__(env)
        # make sure that the environment is an Atari/OpenAI one!
        assert isinstance(env, AtariEnvironment)
        ###########################
        # YOUR IMPLEMENTATION HERE #

        # Declare variables
        self.exp_id = uuid.uuid4().__str__().replace('-', '_')
        self.args = args
        self.env = env
        self.eps_threshold = None
        self.nA = env.action_space.n
        self.action_list = np.arange(self.nA)
        self.reward_list = deque(maxlen=args.window)
        self.max_q_list = deque(maxlen=args.window)
        self.loss_list = deque(maxlen=args.window)
        self.probability_list = np.zeros(env.action_space.n, np.float32)
        self.cur_eps = None
        self.t = 0
        self.ep_len = 0
        self.mode = None
        if self.args.use_pri_buffer:
            self.replay_buffer = PrioritizedBuffer(capacity=self.args.capacity, args=self.args)
        else:
            self.replay_buffer = ReplayBuffer(capacity=self.args.capacity, args=self.args)
        self.position = 0

        self.args.save_dir += f'/{self.exp_id}/'
        os.system(f"mkdir -p {self.args.save_dir}")
        self.meta = DefaultMetaData(fp=open(os.path.join(self.args.save_dir, 'result.csv'), 'w'), args=self.args)
        self.eps_delta = (self.args.eps - self.args.eps_min) / self.args.eps_decay_window

        # Create Policy and Target Networks
        self.policy_net = CNNModel(env, self.args).to(self.args.device)
        self.target_net = CNNModel(env, self.args).to(self.args.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1.5e-4, eps=0.001)
        # Compute Huber loss
        self.loss = F.smooth_l1_loss

        # todo: Support for Multiprocessing. Bug in pytorch - https://github.com/pytorch/examples/issues/370
        # self.policy_net.share_memory()
        # self.target_net.share_memory()

        # Set defaults for networks
        self.policy_net.train()
        self.target_net.eval()
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # if args.test_dqn:
        #     # you can load your model here
        #     ###########################
        #     # YOUR IMPLEMENTATION HERE #
        #     print('loading trained model')
        #     self.load_model()

        if args.use_pri_buffer:
            print('Using priority buffer . . .')
        if args.use_double_dqn:
            print('Using double dqn . . .')

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        ###########################
        pass

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        with torch.no_grad():
            # Fill up probability list equal for all actions
            self.probability_list.fill(self.cur_eps / self.nA)
            # Fetch q from the model prediction
            q, argq = self.policy_net(Variable(self.channel_first(observation))).data.cpu().max(1)
            # Increase the probability for the selected best action
            self.probability_list[argq[0].item()] += 1 - self.cur_eps
            # Use random choice to decide between a random action / best action
            action = torch.tensor([np.random.choice(self.action_list, p=self.probability_list)])
            # if self.args.test_dqn:
            #     action.item()
        ###########################
        return action, q

    def optimize_model(self):
        """
        Function to perform optimization on DL Network
        :return: Loss
        """
        # Return if initial buffer is not filled.
        if len(self.replay_buffer.memory) < self.args.mem_init_size:
            return 0
        self.mode = "Explore"
        if self.args.use_pri_buffer:
            batch_state, batch_action, batch_next_state, batch_reward, batch_done, indices, weights = self.replay_buffer.sample(
                self.args.batch_size)
        else:
            batch_state, batch_action, batch_next_state, batch_reward, batch_done = self.replay_buffer.sample(
                self.args.batch_size)
        policy_max_q = self.policy_net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        if self.args.use_double_dqn:
            policy_ns_max_q = self.policy_net(batch_next_state)
            next_q_value = self.target_net(batch_next_state).gather(1, torch.max(policy_ns_max_q, 1)[1].unsqueeze(
                1)).squeeze(1)
            target_max_q = next_q_value * self.args.gamma * (1 - batch_done)
        else:
            target_max_q = self.target_net(batch_next_state).detach().max(1)[0].squeeze(0) * self.args.gamma * (
                    1 - batch_done)

        # Compute Huber loss
        if self.args.use_pri_buffer:
            loss = self.loss(policy_max_q, batch_reward + target_max_q) * torch.tensor(weights, dtype=torch.float32)
            prios = loss + 1e-5
            loss = loss.mean()
        else:
            loss = self.loss(policy_max_q, batch_reward + target_max_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # Clip rewards between -1 and 1
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        if isinstance(self.replay_buffer, PrioritizedBuffer):
            self.replay_buffer.update_priorities(indices, prios.data.cpu().numpy())

        self.optimizer.step()
        return loss.cpu().detach().numpy()

    def channel_first(self, state):
        """
        The action returned from the environment is nhwc, hence convert to nchw
        :param state: state
        :return: nchw state
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if state.shape[1] == 4:
            return state
        return torch.reshape(state, [1, 84, 84, 4]).permute(0, 3, 1, 2)

    def save_model(self, i_episode):
        """
        Save Model based on condition
        :param i_episode: Episode Number
        """
        if i_episode % self.args.save_freq == 0:
            model_file = os.path.join(self.args.save_dir, f'model_e{i_episode}.th')
            meta_file = os.path.join(self.args.save_dir, f'model_e{i_episode}.meta')
            print(f"Saving model at {model_file}")
            with open(model_file, 'wb') as f:
                torch.save(self.policy_net, f)
            with open(meta_file, 'w') as f:
                self.meta.dump(f)

    def load_model(self):
        """
        Load Model
        :return:
        """
        print(f"Restoring model from {self.args.load_dir} . . . ")
        self.policy_net = torch.load(self.args.load_dir,
                                     map_location=torch.device(self.args.device)).to(self.args.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        if not self.args.test_dqn:
            self.meta.load(open(self.args.load_dir.replace('.th', '.meta')))
            self.t = self.meta.data.step
        else:
            self.cur_eps = 0.01
        print(f"Model successfully restored.")

    def collect_garbage(self, i_episode):
        """
        Collect garbage based on condition
        :param i_episode: Episode Number
        """
        if i_episode % self.args.gc_freq == 0:
            print("Executing garbage collector . . .")
            gc.collect()

    def train(self, persist):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.t = 1
        self.mode = "Random"
        train_start = time.time()
        if not self.args.load_dir == '':
            self.load_model()
        for i_episode in range(1, self.args.max_episodes + 1):
            # Initialize the environment and state
            start_time = time.time()
            state = self.channel_first(self.env.reset())
            self.reward_list.append(0)
            self.loss_list.append(0)
            self.max_q_list.append(0)
            self.ep_len = 0
            done = False

            # Save Model
            self.save_model(i_episode)
            # Collect garbage
            self.collect_garbage(i_episode)

            # Run the game
            while not done:
                # Update the target network, copying all weights and biases in DQN
                if self.t % self.args.target_update == 0:
                    print("Updating target network . . .")
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                # Select and perform an action
                self.cur_eps = max(self.args.eps_min, self.args.eps - self.eps_delta * self.t)
                if self.cur_eps == self.args.eps_min:
                    self.mode = 'Exploit'
                action, q = self.make_action(state)
                next_state, reward, done, _ = self.env.step(action.item())
                self.reward_list[-1] += reward
                self.max_q_list[-1] = max(self.max_q_list[-1], q[0].item())
                next_state = self.channel_first(next_state)
                reward = torch.tensor([reward], device=self.args.device)
                # Store the transition in memory
                self.replay_buffer.push(state, torch.tensor([int(action)]), next_state, reward,
                                        torch.tensor([done], dtype=torch.float32))

                # Increment step and Episode Length
                self.t += 1
                self.ep_len += 1

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                if self.ep_len % self.args.learn_freq == 0:
                    loss = self.optimize_model()
                    self.loss_list[-1] += loss
            self.loss_list[-1] /= self.ep_len

            # Update meta
            self.meta.update(i_episode, self.t, time.time() - start_time, time.time() - train_start,
                             self.ep_len, len(self.replay_buffer.memory), self.cur_eps,
                             self.reward_list[-1], np.mean(self.reward_list),
                             self.max_q_list[-1], np.mean(self.max_q_list),
                             self.loss_list[-1], np.mean(self.loss_list),
                             self.mode)
