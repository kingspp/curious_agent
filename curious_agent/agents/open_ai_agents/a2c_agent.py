"""

### NOTICE ###
DO NOT revise this file

"""
from curious_agent.environments.environment import Environment
from abc import ABCMeta, abstractmethod
from munch import Munch
from typeguard import typechecked
import numpy as np
import os
import logging
import torch
from curious_agent.models import Model
from weakref import ref
from curious_agent import MODULE_CONFIG
import json
import gc
from curious_agent.util import CustomJsonEncoder
from curious_agent.meta.a2c_meta import A2CMetaData
from curious_agent.util import Directories

from curious_agent.agents import Agent
from curious_agent.models.a2c_model import A2CModel
import torch as T
import torch.nn.functional as F
import time

logger = logging.getLogger(__name__)


class A2CAgent(Agent):

    @typechecked
    def __init__(self, env, agent_config: Munch):
        super(A2CAgent, self).__init__(env=env, agent_config=agent_config)
        self.meta = None

        # self.agent_config = agent_config
        self.actor_critic = A2CModel(env=env, config=agent_config, name='actor_critic')

        self.register_models()  # adds the models to ...

        # def preprocess(observation):
        #     return T.Tensor(observation).to(self.state.config.device).permute(2, 0, 1).unsqueeze(0)
        #
        # # This is a permutation of the channel required by PyTorch implementation
        # self.actor_critic.preprocess = preprocess

        self.log_probs = None

    def take_action(self, observation: np.array, test: bool = True):
        probabilities, _ = self.actor_critic.forward(observation)
        probabilities = F.softmax(probabilities)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.log_probs = log_probs

        return action.item()
    
    def learn(self, state, reward, new_state, done):
        self.actor_critic.optimizer.zero_grad()

        _, new_critic_value = self.actor_critic.forward(new_state)
        _, critic_value = self.actor_critic.forward(state)
        reward = T.tensor(reward, dtype=T.float).to(self.state.config.device)

        advantage = reward + self.state.config.gamma*new_critic_value*(1-int(done)) - critic_value

        actor_loss = -self.log_probs * advantage
        critic_loss = advantage**2

        loss = actor_loss + critic_loss
        loss.backward()

        self.actor_critic.optimizer.step()

        return loss

    @typechecked
    def train(self, persist: bool, run: int = -1, checkpoint: int = -1):
        # the initialization branches mentioned above
        if not persist:  # Starting
            self.state.i_episode = 0
            self.state.num_episodes = 70000
            self.meta = A2CMetaData(fp=open(os.path.join(MODULE_CONFIG.BaseConfig.BASE_DIR, 'agent_stats.csv'), 'w'),
                                    args=self.state.config)
            pass  # custom startup initialization
        else:  # Continuing
            self.load_model()
            pass  # custom continuing initialization

        while self.state.i_episode < self.state.num_episodes:
            start_time = time.time()
            self.state.done = False
            self.state.score = 0
            self.state.observation = self.env.reset()

            while not self.state.done:
                action = self.take_action(self.state.observation)
                new_observation, reward, self.state.done, info = self.env.step(action)
                self.state.loss = self.learn(self.state.observation, reward, new_observation, self.state.done)
                self.state.observation = new_observation
                self.state.score += reward
            
            self.save(self.state.i_episode)            
            self.collect_garbage(self.state.i_episode)
            self.state.i_episode += 1
            
            # Update meta
            self.meta.update_episode(self.state.i_episode, time.time() - start_time, self.state.score, self.state.loss.item())
    
    def load_model(self):
        """
        Load Model
        :return:
        """
        if self.state.config.load_dir == '':
            return
        
        logger.info(f"Restoring model from {self.state.config.load_dir} . . . ")
        self.actor_critic = torch.load(self.state.config.load_dir,
                                     map_location=torch.device(self.state.config.device)).to(self.state.config.device)
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
