#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import random
import numpy as np
from collections import deque
import os
import torch
from torch import nn, tensor
import torch.nn.functional as F
import torch.optim as optim
from curious_agent.agents.agent import Agent
from curious_agent.models.dqn_model import DQN
from curious_agent.models.icm_model import ICM
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from munch import Munch
from curious_agent import MODULE_CONFIG
from curious_agent.meta.icm_meta_v1 import ICMMetaDataV1
import time

"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


def generate_onehot(index, num_actions):
    return [1 if i == index else 0 for i, x in enumerate(range(num_actions))]


class Agent_DQN(Agent):
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

        super(Agent_DQN, self).__init__(env=env, agent_config=agent_config)

        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.state = Munch({**self.state,
                            "step": 0,
                            "num_actions": env.action_space.n,
                            "metrics_capture_window": agent_config.metrics_capture_window,
                            "replay_size": agent_config.replay_size,
                            "position": 0,
                            "total_num_steps": agent_config.total_num_steps,
                            "episodes": agent_config.episodes,
                            "gamma": agent_config.gamma,
                            "learning_rate": agent_config.learning_rate,
                            "initial_epsilon": agent_config.initial_epsilon,
                            "final_epsilon": agent_config.final_epsilon,
                            "epsilon": agent_config.initial_epsilon,
                            "steps_to_explore": agent_config.steps_to_explore,
                            "network_update_interval": agent_config.network_update_interval,
                            "network_train_interval": agent_config.network_train_interval,
                            "batch_size": agent_config.batch_size,
                            "mode": "Random",
                            "state_counter_while_testing": 0,
                            "beta": agent_config.beta,
                            "lambda_val": agent_config.lambda_val,
                            "eta": agent_config.eta
                            })
        # self.run_name = agent_config.run_name
        # self.state.model_save_path = agent_config.model_save_path
        # self.state.model_save_interval = agent_config.model_save_interval
        # self.log_path = agent_config.log_path
        # self.tensorboard_summary_path = agent_config.tensorboard_summary_path
        self.is_cuda_available = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.is_cuda_available else "cpu")
        # self.state.model_test_path = agent_config.model_test_path
        # self.state.step = 0

        # Environment and network parameters
        # self.env = env
        # self.state.num_actions = env.action_space.n
        self.action_list = np.arange(self.state.num_actions)
        self.input_shape = [4, 84, 84]
        # self.state.metrics_capture_window = agent_config.metrics_capture_window
        # self.state.replay_size = agent_config.replay_size
        self.replay_memory = []
        # self.state.position = 0
        # self.state.total_num_steps = agent_config.total_num_steps
        # self.state.episodes = agent_config.episodes
        # self.state.gamma = agent_config.gamma
        # self.state.learning_rate = agent_config.learning_rate
        # self.state.initial_epsilon = agent_config.initial_epsilon
        # self.state.final_epsilon = agent_config.final_epsilon
        # self.state.epsilon = self.state.initial_epsilon
        # self.state.steps_to_explore = agent_config.steps_to_explore
        self.state.epsilon_step = (self.state.initial_epsilon - self.state.final_epsilon) / self.state.steps_to_explore

        # self.state.network_update_interval = agent_config.network_update_interval
        # self.state.network_train_interval = agent_config.network_train_interval

        self.last_n_rewards = deque([], self.state.metrics_capture_window)
        self.start_to_learn = agent_config.start_to_learn
        self.ddqn = agent_config.ddqn
        self.use_icm = agent_config.use_icm
        self.intrinsic_episode_reward = []
        self.last_n_intrinsic_rewards = deque([], self.state.metrics_capture_window)

        # self.state.batch_size = agent_config.batch_size
        # self.state.mode = "Random"
        # self.state.state_counter_while_testing = 0
        self.q_network = DQN(env=env, args=self.state.config).to(self.device)
        self.target_network = DQN(env=env, args=self.state.config).to(self.device)
        self.loss_function = F.smooth_l1_loss
        self.optimiser = optim.Adam(self.q_network.parameters(), lr=agent_config.learning_rate)
        self.probability_list = np.zeros(env.action_space.n, np.float32)
        self.q_network.train()
        self.target_network.eval()

        self.icm_model = ICM(env=env, num_actions=self.state.num_actions, args=self.state.config).to(self.device)

        self.inverse_loss_fn = nn.CrossEntropyLoss()
        self.forward_loss_fn = nn.MSELoss()

        # self.state.beta = agent_config.beta
        # self.state.lambda_val = agent_config.lambda_val
        # self.state.eta = agent_config.eta

        # create necessary paths
        # self.create_dirs()
        self.meta = None

        if agent_config.test_dqn:
            print('loading trained model')
            self.q_network.load_state_dict(torch.load(self.state.model_test_path, map_location=self.device))

        # self.log_file = open(self.state.model_save_path + '/' + self.run_name + '.log', 'w') if not agent_config.test_dqn else None

        # Set target_network weight
        self.target_network.load_state_dict(self.q_network.state_dict())

        # self.writer = SummaryWriter(agent_config.tensorboard_summary_path)

    def create_dirs(self):
        paths = [self.state.model_save_path, self.tensorboard_summary_path]
        [os.makedirs(path) for path in paths if not os.path.exists(path)]

    def take_action(self, observation, test=True, **kwargs):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        self.init_game_setting()
        with torch.no_grad():
            if test:
                # if kwargs["state_count"] < 5000:
                action = torch.argmax(
                    self.q_network(tensor(observation).unsqueeze(0).permute(0, 3, 1, 2).float()).detach())
                return action.item()
            # Fill up probability list equal for all actions
            self.probability_list.fill(self.state.epsilon / self.state.num_actions)
            # Fetch q from the model prediction
            q, argq = self.q_network(tensor(observation).float()).data.cpu().max(1)
            # Increase the probability for the selected best action
            self.probability_list[argq[0].item()] += 1 - self.state.epsilon
            # Use random choice to decide between a random action / best action
            action = torch.tensor([np.random.choice(self.action_list, p=self.probability_list)])
        return action.item(), q

    def init_game_setting(self):
        """

        Testing function will call this function at the beginning of new game
        Put anything you want to initialize if necessary

        """
        self.state.state_counter_while_testing += 1

    def push(self, transition_tuple):
        """ You can add additional arguments as you need.
        Push new data to buffer and remove the old one if the buffer is full.

        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        if len(self.replay_memory) < self.state.replay_size:
            self.replay_memory.append(None)
        self.replay_memory[self.state.position] = transition_tuple
        self.state.position = (self.state.position + 1) % self.state.replay_size

    def optimize_network(self):

        if len(self.replay_memory) < self.state.replay_size:
            return 0

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, self.state.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = map(
                lambda x: Variable(torch.cat(x, 0)), zip(*minibatch))

        if self.use_icm:
            # first calculate loss values
            # 1. discounted reward
            # 2. inverse model loss
            # 3. forward model loss
            onehot_action_batch = tensor([generate_onehot(x, self.state.num_actions) for x in action_batch]).float()
            encoded_next_state_batch, predicted_next_state_batch, predicted_action_batch = self.icm_model([state_batch,
                                                                                                           next_state_batch,
                                                                                                           onehot_action_batch])
            loss_inverse = self.inverse_loss_fn(predicted_action_batch, action_batch)
            loss_forward = self.forward_loss_fn(predicted_next_state_batch, encoded_next_state_batch)

            intrinsic_reward = self.state.eta * loss_forward
            discounted_reward = self.state.lambda_val * (intrinsic_reward + reward_batch)

            self.intrinsic_episode_reward.append(intrinsic_reward.detach().cpu().numpy())
            q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
            target_values = self.target_network(next_state_batch)
            if self.ddqn:
                best_actions = torch.argmax(self.q_network(next_state_batch), dim=-1)
                target_values = target_values.gather(1, tensor(best_actions).unsqueeze(1)).squeeze(1)
            else:
                target_values = target_values.max(1)[0].squeeze(0)
            target_values = target_values * self.state.gamma * (1 - terminal_batch)
            q_loss = self.loss_function(q_values, discounted_reward + target_values)
            loss = q_loss + loss_forward + loss_inverse
        # else:
        #     # Normal Deep-Q-Learning agent
        #     q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        #     target_values = self.target_network(next_state_batch)
        #     if self.ddqn:
        #         best_actions = torch.argmax(self.q_network(next_state_batch), dim=-1)
        #         target_values = target_values.gather(1, tensor(best_actions).unsqueeze(1)).squeeze(1)
        #     else:
        #         target_values = target_values.max(1)[0].squeeze(0)
        #     target_values = target_values * self.state.gamma * (1 - terminal_batch)
        #     loss = self.loss_function(q_values, reward_batch + target_values)

        self.optimiser.zero_grad()
        loss.backward()
        # for param in self.q_network.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimiser.step()
        return loss.cpu().detach().numpy()

    def train(self, persist: bool = False, run: int = -1, checkpoint: int = -1):
        """
        Implement your training algorithm here
        """
        self.meta = ICMMetaDataV1(fp=open(os.path.join(MODULE_CONFIG.BaseConfig.BASE_DIR, 'agent_stats.csv'), 'w'),
                                  args=self.state.config)
        train_start = time.time()
        for episode in range(self.state.episodes):
            start_time = time.time()
            state = self.env.reset()
            state = torch.reshape(tensor(state, dtype=torch.float32), [1, 84, 84, 4]).permute(0, 3, 1, 2).to(
                    self.device)
            done = False
            episode_reward = []
            episode_loss = []

            # save network
            # if episode % self.state.model_save_interval == 0:
            #     save_path = self.state.model_save_path + '/' + self.run_name + '_' + str(episode) + '.pt'
            #     torch.save(self.q_network.state_dict(), save_path)
            #     print('Successfully saved: ' + save_path)

            # Save Model
            self.save(episode)
            # Collect garbage
            # To Do Later

            while not done:

                # update target network
                if self.state.step % self.state.network_update_interval == 0:
                    print('Updating target network')
                    self.target_network.load_state_dict(self.q_network.state_dict())

                if self.state.step > len(self.replay_memory):
                    self.state.epsilon = max(self.state.final_epsilon,
                                             self.state.initial_epsilon - self.state.epsilon_step * self.state.step)
                    if self.state.epsilon > self.state.final_epsilon:
                        self.state.mode = 'Explore'
                    else:
                        self.state.mode = 'Exploit'

                action, q = self.take_action(state, test=False, state_count=0)
                next_state, reward, done, _ = self.env.step(action)

                next_state = torch.reshape(tensor(next_state, dtype=torch.float32), [1, 84, 84, 4]).permute(0, 3, 1,
                                                                                                            2).to(
                        self.device)
                self.push((state, torch.tensor([int(action)]), torch.tensor([reward], device=self.device), next_state,
                           torch.tensor([done], dtype=torch.float32)))
                episode_reward.append(reward)
                self.state.step += 1
                state = next_state

                # train network
                if self.state.step >= self.start_to_learn and self.state.step % self.state.network_train_interval == 0:
                    loss = self.optimize_network()
                    episode_loss.append(loss)

                if done:
                    # print('Episode:', episode, ' | Steps:', self.state.step, ' | Eps: ', self.state.epsilon,
                    #       ' | Reward: ',
                    #       sum(episode_reward),
                    #       ' | Avg Reward: ', np.mean(self.last_n_rewards), ' | Loss: ',
                    #       np.mean(episode_loss), ' | Intrinsic Reward: ', sum(self.intrinsic_episode_reward),
                    #       ' | Avg Intrinsic Reward: ', np.mean(self.last_n_intrinsic_rewards),
                    #       ' | Mode: ', self.state.mode)
                    # print('Episode:', episode, ' | Steps:', self.state.step, ' | Eps: ', self.state.epsilon,
                    #       ' | Reward: ',
                    #       sum(episode_reward),
                    #       ' | Avg Reward: ', np.mean(self.last_n_rewards), ' | Loss: ',
                    #       np.mean(episode_loss), ' | Intrinsic Reward: ', sum(self.intrinsic_episode_reward),
                    #       ' | Avg Intrinsic Reward: ', np.mean(self.last_n_intrinsic_rewards),
                    #       ' | Mode: ', self.state.mode, file=self.log_file)
                    # self.log_summary(episode, episode_loss, episode_reward)
                    self.last_n_rewards.append(sum(episode_reward))
                    self.last_n_intrinsic_rewards.append(sum(self.intrinsic_episode_reward))
                    self.meta.update_episode(episode, self.state.step, self.state.epsilon,
                                             sum(episode_reward), np.mean(self.last_n_rewards),
                                             np.mean(episode_loss), sum(self.intrinsic_episode_reward),
                                             np.mean(self.last_n_intrinsic_rewards), self.state.mode)

                    episode_reward.clear()
                    episode_loss.clear()
                    self.intrinsic_episode_reward.clear()
