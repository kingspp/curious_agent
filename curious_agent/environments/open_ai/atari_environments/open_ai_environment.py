import gym
import numpy as np
from curious_agent.environments.open_ai.atari_environments.atari_wrapper import make_wrap_atari
from curious_agent.environments.environment import Environment


class AtariEnvironment(Environment):
    def __init__(self, args, atari_wrapper=False, test=False):
        if atari_wrapper:
            clip_rewards = not test
            self.env = make_wrap_atari(args['env_name'], clip_rewards)
        else:
            self.env = gym.make(args['env_name'])
        self.args=args
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        super().__init__()
        
    def seed(self, seed):
        '''
        Control the randomness of the environment
        '''
        self.env.seed(seed)

    def reset(self):
        '''
        When running dqn:
            observation: np.array
                stack 4 last frames, shape: (84, 84, 4)

        When running pg:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)
        '''
        observation = self.env._reset()

        return np.array(observation)


    def step(self,action):
        '''
        When running dqn:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
            reward: int
                wrapper clips the reward to {-1, 0, 1} by its sign
                we don't clip the reward when testing
            done: bool
                whether reach the end of the episode?

        When running pg:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)
            reward: int
                if opponent wins, reward = +1 else -1
            done: bool
                whether reach the end of the episode?
        '''
        if not self.env.action_space.contains(action):
            raise ValueError('Ivalid action!!')

        observation, reward, done, info = self.env.step(action)

        return np.array(observation), reward, done, info


    def get_action_space(self):
        return self.action_space


    def get_observation_space(self):
        return self.observation_space


    def get_random_action(self):
        return self.action_space.sample()
