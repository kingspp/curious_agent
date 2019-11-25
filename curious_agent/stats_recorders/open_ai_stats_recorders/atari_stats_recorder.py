"""

### NOTICE ###
You DO NOT need to upload this file

"""

import numpy as np
import time
from cv2 import VideoWriter, VideoWriter_fourcc
from gym.wrappers import Monitor
import logging



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

    def load(self, location):
        self.agent.load(location)


    def record(self, output):
        logger.info("Started recording. . .")
        # self.env.env = Monitor(self.env.env, output + "/video", force=True, write_upon_reset=True)
        rewards = []
        self.env.seed(seed)
        start_time = time.time()
        height, width, channels = 0, 0, 0
        for i in range(self.episodes_number):
            state = self.env.reset()
            # grab the dimensions off the first frame
            height, width, channels = self.env.env.render(mode='rgb_array').shape

            self.agent.init_game_setting()
            done = False
            episode_reward = 0.0

            # playing one game
            frames = [state]
            while not done:
                # get the image instead of displaying it
                img = self.env.env.render(mode='rgb_array')
                action = self.agent.take_action(state, test=True)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
                # store the frame
                frames.append(img)

            rewards.append(episode_reward)

        self.env.env.close()

        # turn the frames list into a video
        four_cc = VideoWriter_fourcc(*'h264')
        frames_per_second = 24
        video = VideoWriter(output + '/output.mp4', four_cc, float(frames_per_second), (width, height))
        for frame in frames:
            video.write(frame)
        video.release()

        print('Run %d episodes' % self.episodes_number)
        print('Mean:', np.mean(rewards))
        print('rewards', rewards)
        print('running time', time.time() - start_time)
        logger.info("Stopped the testing/recording. . .")
