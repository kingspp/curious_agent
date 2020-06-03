# -*- coding: utf-8 -*-
"""
@created on: 12/6/19,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

# !/usr/bin/env python3
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
from curious_agent.environments.open_ai.atari.atari_environment import AtariEnvironment
from curious_agent.buffers import PrioritizedBuffer, ReplayBuffer
from curious_agent.meta.icm_meta import ICMMetaData
from munch import Munch
from curious_agent.models.icm_model import ICM
import logging
from curious_agent import MODULE_CONFIG
from torch import nn
from torch import tensor
from curious_agent.util.generic_utils import generate_onehot

logger = logging.getLogger(__name__)

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class DQNAgentICM(Agent):

    def __init__(self, env, agent_config: Munch):
        """
        Initialize everything you need here.
        For example:
            paramters for neural network
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(DQNAgentICM, self).__init__(env=env, agent_config=agent_config)
        # make sure that the environment is an Atari/OpenAI one!
        assert isinstance(env, AtariEnvironment)
        # Declare primitive variables
        self.state = Munch({**self.state,
                            "num_actions": env.action_space.n,
                            "cur_eps": None,
                            "t": 0,
                            "ep_len": 0,
                            "mode": None,
                            "position": 0,
                            })

        self.state.config.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reward_list = deque(maxlen=agent_config.window)
        self.intrinsic_reward_list = deque(maxlen=agent_config.window)
        self.episodic_intrinsic_reward_list = deque(maxlen=agent_config.window)
        self.max_q_list = deque(maxlen=agent_config.window)
        self.loss_list = deque(maxlen=agent_config.window)
        self.probability_list = np.zeros(env.action_space.n, np.float32)
        self.action_list = np.arange(env.action_space.n)

        self.state.eps_delta = (self.state.config.eps - self.state.config.eps_min) / self.state.config.eps_decay_window

        if self.state.config.use_pri_buffer:
            self.replay_buffer = PrioritizedBuffer(capacity=self.state.config.capacity, args=self.state.config)
        else:
            self.replay_buffer = ReplayBuffer(capacity=self.state.config.capacity, args=self.state.config)

        self.env = env
        self.meta = None
        # Create Policy and Target Networks
        self.policy_net = CNNModel(env, self.state.config).to(self.state.config.device)
        self.target_net = CNNModel(env, self.state.config).to(self.state.config.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1.5e-4, eps=0.001)
        # Compute Huber loss
        self.loss = F.smooth_l1_loss

        # Set defaults for networks
        self.policy_net.train()
        self.target_net.eval()
        self.target_net.load_state_dict(self.policy_net.state_dict())

        if agent_config.use_pri_buffer:
            logger.info('Using priority buffer . . .')
        if agent_config.use_double_dqn:
            logger.info('Using double dqn . . .')

        self.icm_model = ICM(self.state.num_actions, env=env, args=self.state.config).to(self.state.config.device)

        self.inverse_loss_fn = nn.CrossEntropyLoss()
        self.forward_loss_fn = nn.MSELoss()
        self.beta = self.state.config.beta
        self.lambda_val = self.state.config.lambda_val
        self.eta = self.state.config.eta

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

    def take_action(self, observation, test=False):
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
            self.probability_list.fill(self.state.cur_eps / self.state.num_actions)
            # Fetch q from the model prediction
            q, argq = self.policy_net(Variable(self.channel_first(observation))).data.cpu().max(1)
            # Increase the probability for the selected best action
            self.probability_list[argq[0].item()] += 1 - self.state.cur_eps
            # Use random choice to decide between a random action / best action
            action = torch.tensor([np.random.choice(self.action_list, p=self.probability_list)])
            if test:
                return action.item()
        ###########################
        return action, q

    def optimize_model(self):
        """
        Function to perform optimization on DL Network
        :return: Loss
        """
        # Return if initial buffer is not filled.
        if len(self.replay_buffer.memory) < self.state.config.mem_init_size:
            return 0
        self.state.mode = "Explore"
        if self.state.config.use_pri_buffer:
            batch_state, batch_action, batch_next_state, batch_reward, batch_done, indices, weights = self.replay_buffer.sample(
                    self.state.config.batch_size)
        else:
            batch_state, batch_action, batch_next_state, batch_reward, batch_done = self.replay_buffer.sample(
                    self.state.config.batch_size)
        policy_max_q = self.policy_net(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        if self.state.config.use_double_dqn:
            policy_ns_max_q = self.policy_net(batch_next_state)
            next_q_value = self.target_net(batch_next_state).gather(1, torch.max(policy_ns_max_q, 1)[1].unsqueeze(
                    1)).squeeze(1)
            target_max_q = next_q_value * self.state.config.gamma * (1 - batch_done)
        else:
            target_max_q = self.target_net(batch_next_state).detach().max(1)[0].squeeze(0) * self.state.config.gamma * (
                    1 - batch_done)

        # Adding ICM components
        onehot_action_batch = tensor([generate_onehot(x.item(), self.state.num_actions) for x in batch_action]).float()
        icm_input = (batch_state, batch_next_state, onehot_action_batch)
        encoded_next_state_batch, predicted_next_state_batch, predicted_action_batch = self.icm_model(icm_input)

        loss_inverse = self.inverse_loss_fn(predicted_action_batch, batch_action)
        loss_forward = self.forward_loss_fn(predicted_next_state_batch, encoded_next_state_batch)

        intrinsic_reward = self.eta * loss_forward
        discounted_reward = self.lambda_val * (intrinsic_reward + batch_reward)
        self.episodic_intrinsic_reward_list.append(intrinsic_reward.detach().cpu().numpy())

        # Compute Huber loss
        if self.state.config.use_pri_buffer:
            loss = self.loss(policy_max_q, discounted_reward + target_max_q) * torch.tensor(weights,
                                                                                            dtype=torch.float32)
            prios = loss + 1e-5
            q_loss = loss.mean()
        else:
            q_loss = self.loss(policy_max_q, discounted_reward + target_max_q)

        loss = q_loss + loss_forward + loss_inverse

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

    def load_model(self):
        """
        Load Model
        :return:
        """
        logger.info(f"Restoring model from {self.state.config.load_dir} . . . ")
        self.policy_net = torch.load(self.state.config.load_dir,
                                     map_location=torch.device(self.state.config.device)).to(self.state.config.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        if not self.state.config.test_dqn:
            self.meta.load(open(self.state.config.load_dir.replace('.th', '.meta')))
            self.state.t = self.meta.data.step
        else:
            self.state.cur_eps = 0.01
        logger.info(f"Model successfully restored.")

    def collect_garbage(self, i_episode):
        """
        Collect garbage based on condition
        :param i_episode: Episode Number
        """
        if i_episode % self.state.config.gc_freq == 0:
            logger.info("Executing garbage collector . . .")
            gc.collect()

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.meta = ICMMetaData(fp=open(os.path.join(MODULE_CONFIG.BaseConfig.BASE_DIR, 'agent_stats.csv'), 'w'),
                                args=self.state.config)
        self.state.t = 1
        self.state.mode = "Random"
        train_start = time.time()
        if not self.state.config.load_dir == '':
            self.load_model()
        for i_episode in range(1, self.state.config.max_episodes + 1):
            # Initialize the environment and state
            start_time = time.time()
            state = self.channel_first(self.env.reset())
            self.reward_list.append(0)
            self.loss_list.append(0)
            self.reward_list.append(0)
            self.state.ep_len = 0
            done = False

            # Save Model
            self.save(i_episode)
            # Collect garbage
            self.collect_garbage(i_episode)

            # Run the game
            while not done:
                # Update the target network, copying all weights and biases in DQN
                if self.state.t % self.state.config.target_update == 0:
                    logger.info("Updating target network . . .")
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                # Select and perform an action
                self.state.cur_eps = max(self.state.config.eps_min,
                                         self.state.config.eps - self.state.eps_delta * self.state.t)
                if self.state.cur_eps == self.state.config.eps_min:
                    self.state.mode = 'Exploit'
                action, q = self.take_action(state)
                next_state, reward, done, _ = self.env.step(action.item())
                self.reward_list[-1] += reward
                self.reward_list[-1] = max(self.reward_list[-1], q[0].item())
                next_state = self.channel_first(next_state)
                reward = torch.tensor([reward], device=self.state.config.device)
                # Store the transition in memory
                self.replay_buffer.push(state, torch.tensor([int(action)]), next_state, reward,
                                        torch.tensor([done], dtype=torch.float32))
                self.meta.update_step(self.state.t, self.state.cur_eps, self.reward_list[-1],
                                      self.reward_list[-1],
                                      self.loss_list[-1], self.state.config.lr,
                                      self.episodic_intrinsic_reward_list[-1] if len(
                                              self.episodic_intrinsic_reward_list) > 0 else 0)

                # Increment step and Episode Length
                self.state.t += 1
                self.state.ep_len += 1

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                if self.state.ep_len % self.state.config.learn_freq == 0:
                    loss = self.optimize_model()
                    self.loss_list[-1] += loss
            self.loss_list[-1] /= self.state.ep_len

            # Collecting episode level intrinsic reward
            self.intrinsic_reward_list.append(sum(self.episodic_intrinsic_reward_list))
            self.episodic_intrinsic_reward_list.clear()

            # Update meta
            self.meta.update_episode(i_episode, self.state.t, time.time() - start_time, time.time() - train_start,
                                     self.state.ep_len, len(self.replay_buffer.memory), self.state.cur_eps,
                                     self.reward_list[-1], np.mean(self.reward_list),
                                     self.reward_list[-1], np.mean(self.reward_list),
                                     self.loss_list[-1], np.mean(self.loss_list),
                                     self.state.mode, self.state.config.lr,
                                     np.mean(self.intrinsic_reward_list) if len(self.intrinsic_reward_list) > 0 else 0)
