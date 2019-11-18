from collections import namedtuple
import json
from curious_agent.util.custom_json_encoder import CustomJsonEncoder
import logging
from humanize import naturaltime
from curious_agent import MODULE_CONFIG
import os

logger = logging.getLogger(__name__)


class DefaultMetaData(object):
    """
    Medata for model monitor and restore purpose
    """

    def __init__(self, fp, args):
        self.episode_template = namedtuple('EpisodeData',
                                           (
                                               "episode", "step", "time", "time_elapsed", "ep_len", "buffer_len",
                                               "epsilon",
                                               "reward", "avg_reward", "max_q", "max_avg_q", "loss", "avg_loss", "mode",
                                               "lr"))
        self.step_template = namedtuple('StepData', ("step", "epsilon", "reward", "max_q", "loss", "lr"))
        self.fp = fp
        self.episode_data = None
        self.step_data = None
        self.args = args
        if MODULE_CONFIG.BaseConfig.TENSORBOARD_SUMMARIES:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(MODULE_CONFIG.BaseConfig.PATH_GRAPHS)

    def update_step(self, *args):
        self.step_data = self.step_template(*args)
        if MODULE_CONFIG.BaseConfig.TENSORBOARD_SUMMARIES:
            self.writer.add_scalar('step/epsilon', self.step_data.epsilon, self.step_data.step)
            self.writer.add_scalar('step/learning_rate', self.step_data.lr, self.step_data.step)
            self.writer.add_scalar('step/reward', self.step_data.reward, self.step_data.step)
            self.writer.add_scalar('step/max_q', self.step_data.max_q, self.step_data.step)
            self.writer.add_scalar('step/loss', self.step_data.loss, self.step_data.step)

    def update_episode(self, *args):
        """
        Update metadata
        :param args: args
        """
        self.episode_data = self.episode_template(*args)
        if self.episode_data.episode % self.args.disp_freq == 0:
            print(
                f"E: {self.episode_data.episode} | M: {self.episode_data.buffer_len} |  Step: {self.episode_data.step} "
                f"| T: {self.episode_data.time:.2f} | Len: {self.episode_data.ep_len} | EPS: {self.episode_data.epsilon:.5f} "
                f"| LR: {self.episode_data.lr:.7f} | R: {self.episode_data.reward} | AR: {self.episode_data.avg_reward:.3f} "
                f"| MAQ:{self.episode_data.max_avg_q:.2f} "
                f"| L: {self.episode_data.loss:.2f} | AL: {self.episode_data.avg_loss:.4f} | Mode: {self.episode_data.mode} "
                f"| ET: {naturaltime(self.episode_data.time_elapsed)}")
        if MODULE_CONFIG.BaseConfig.TENSORBOARD_SUMMARIES:
            self.writer.add_scalar('episode/epsilon', self.episode_data.epsilon, self.episode_data.episode)
            self.writer.add_scalar('episode/steps', self.episode_data.step, self.episode_data.episode)
            self.writer.add_scalar('episode/learning_rate', self.episode_data.lr, self.episode_data.episode)
            self.writer.add_scalar('episode/avg_reward', self.episode_data.avg_reward, self.episode_data.episode)
            self.writer.add_scalar('episode/avg_max_q', self.episode_data.max_avg_q, self.episode_data.episode)
            self.writer.add_scalar('episode/avg_loss', self.episode_data.avg_loss, self.episode_data.episode)

        self.fp.write(self.episode_data._asdict().values().__str__().replace('odict_values([', '').replace('])', '\n'))

    def load(self, f):
        """
        Load Metadata
        :param f: File Pointer
        :return:
        """
        self.episode_data = self.episode_data(*json.load(f).values())

    def dump(self, f):
        """
        JSONify metadata
        :param f: file pointer
        """
        json.dump(self.episode_data._asdict(), f, cls=CustomJsonEncoder, indent=2)
