"""

### NOTICE ###
You DO NOT need to upload this file

"""

import numpy as np
import time
from cv2 import VideoWriter, VideoWriter_fourcc
from gym.wrappers import Monitor
import logging
import json

from curious_agent.environments.environment import Environment
# from curious_agent.agents.agent import Agent

from curious_agent.agents.agent import Agent
from curious_agent.stats_recorders.stats_recorder import StatsRecorder
from curious_agent.environments.open_ai.atari.atari_environment import AtariEnvironment
from typeguard import typechecked

logger = logging.getLogger(__name__)

seed = 11037


class AtariEnvStatsRecorder(StatsRecorder):
    """Abstract class that contains the definition of the abstract load and record functions that provide an interface
    to produce statistics from an environment given an agent and an environment.

    """

    @typechecked
    def __init__(self, agent: Agent, env: AtariEnvironment, episodes_number: int):
        """

        :param agent: a testing agent (should not be the same agent that is being trained)
        :param env: a testing environment (should not be the same environment that is being used for training)
        """
        self.agent = agent
        self.env = env
        # self.env.env = Monitor(self.env.env, './output', force=True)
        self.episodes_number = episodes_number

    @typechecked
    def load(self, location: str):
        self.agent.load(location)

    @typechecked
    def record(self, output: str):
        logger.debug("Started recording. . .")
        def return_true(idx):
            return True
        self.env.env = Monitor(self.env.env, output + "_video", force=True, video_callable=return_true, mode='evaluation')
        rewards = []
        self.env.seed(seed)
        start_time = time.time()
        height, width, channels = 0, 0, 0
        for i in range(self.episodes_number):
            state = self.env.reset()
            # grab the dimensions off the first frame
            height, width, channels = self.env.env.render(mode='rgb_array').shape

            done = False
            episode_reward = 0.0

            # playing one game
            while not done:
                # get the image instead of displaying it
                # self.env.env.render(mode='rgb_array')
                action = self.agent.take_action(state, test=True)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward

            rewards.append(episode_reward)

        # self.env.env.close()

        # turn the frames list into a video
        # four_cc = VideoWriter_fourcc(*'MP42')
        # frames_per_second = 24
        # video = VideoWriter(output + '/output.avi', four_cc, float(frames_per_second), (width, height))
        # for frame in frames:
        #     video.write(frame)
        # logger.info("Releasing the video. . .")
        # video.release()

        # logger.debug('Run ' + str(self.episodes_number) + ' episodes')
        # logger.debug('Mean: ' + str(np.mean(rewards)))
        # logger.debug('running time: ' + str(time.time() - start_time))
        stats = {
            "episodes_tested": self.episodes_number,
            "mean_reward": np.mean(rewards),
            "elapsed_time": time.time() - start_time
        }
        logger.debug("Stopped the testing/recording. . .")
        logger.info(
            f"episodes tested: {stats['episodes_tested']} | Mean Reward: {stats['mean_reward']} |  ET: {stats['elapsed_time']} ")
        json.dump(stats, open(output + "_stats.json", 'w'), indent=2)
